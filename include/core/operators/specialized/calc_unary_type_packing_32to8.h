#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_32to8_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_32to8_H

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/type_packing.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/type_helper.h>
#include <core/operators/interfaces/calc.h>

namespace morphstore {

using namespace vectorlib;

template<
        template<class, int> class t_operator
>                                                                    
struct calc_unary_t<t_operator, vectorlib::sse<vectorlib::v128<uint64_t>>, type_packing_f<uint8_t >, type_packing_f<uint32_t >>  
    {
    using t_vector_extension = vectorlib::sse<vectorlib::v128<uint64_t>>;
    using t_in_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype;
    using t_out_ve = typename TypeHelper<t_vector_extension, uint8_t>::newbasetype; 
    
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_in_ve, in_)
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_out_ve, out_)
    MSV_CXX_ATTRIBUTE_FORCE_INLINE

                                    
    static column<type_packing_f<uint8_t >> const * apply(
            column<type_packing_f<uint32_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {        
        size_t const inDataCount = inDataCol->get_count_values();
        in_base_t const * inData = inDataCol->get_data();

        auto outPosCol = new column<type_packing_f<uint8_t >>( 
                bool(outPosCountEstimate)
                // use given estimate                          
                ? get_size_max_byte_any_len<type_packing_f<uint8_t >>(outPosCountEstimate)
                // use pessimistic estimate                    
                : get_size_max_byte_any_len<type_packing_f<uint8_t >>(inDataCount)
        );

        out_base_t * outPos = outPosCol->get_data();
        out_base_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint64_t >::m_BlockSize;
        size_t const vectorMultiplier = blockSize / out_vector_element_count::value;
        size_t const vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t const remainderCount = inDataCount % blockSize;
        // std::cout<< "Vectorcount: " << vectorCount << std::endl;
        // std::cout<< "remainderCount: " << remainderCount << std::endl;
        // std::cout<< "blockSize: " << blockSize << std::endl;

        in_vector_t const predicateVector = set1<t_in_ve, in_vector_base_t_granularity::value>(val);
        for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part

            //first add/substract datavector and predicateVector (32 bit) and then compress the result to 8 bit
            //create four resultVectorCompressed vectors and combine them to one via vectorlib::bitwise_or
            //store this one final vector
            in_vector_t dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData); //load 4 values for sse
            in_vector_t result =
                t_operator<t_in_ve, in_vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            in_vector_t resultVectorCompressed1 = _mm_shuffle_epi8(result, _mm_set_epi8(127,127,127,127,127,127,127,127,
               127,127,127,127,12,8,4,0));          
            inData += in_vector_element_count::value;

            dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData); //load 4 values for sse
            result =
                t_operator<t_in_ve, in_vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            in_vector_t resultVectorCompressed2 = _mm_shuffle_epi8(result, _mm_set_epi8(127,127,127,127,127,127,127,127,
               12,8,4,0,127,127,127,127));          
            inData += in_vector_element_count::value;

            dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData); //load 4 values for sse
            result =
                t_operator<t_in_ve, in_vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            in_vector_t resultVectorCompressed3 = _mm_shuffle_epi8(result, _mm_set_epi8(127,127,127,127,12,8,4,0,
               127,127,127,127,127,127,127,127));  
            inData += in_vector_element_count::value;

            dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData); //load 4 values for sse
            result =
                t_operator<t_in_ve, in_vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            in_vector_t resultVectorCompressed4 = _mm_shuffle_epi8(result, _mm_set_epi8(12,8,4,0,127,127,127,127,
               127,127,127,127,127,127,127,127));  

            in_vector_t resultVectorCompressed = bitwise_or<t_in_ve>(resultVectorCompressed1,resultVectorCompressed2);
            resultVectorCompressed = bitwise_or<t_in_ve>(resultVectorCompressed,  resultVectorCompressed3);
            resultVectorCompressed = bitwise_or<t_in_ve>(resultVectorCompressed,  resultVectorCompressed4);
            vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, resultVectorCompressed); //store 16 values

            outPos += out_vector_element_count::value;
            inData += in_vector_element_count::value;
            //std::cout<< "in compressed "  << std::endl;
        }        

        //create padding
        uint64_t * outPosUncompr = reinterpret_cast< uint64_t *>(outPosCol->create_data_uncompr_start(reinterpret_cast< uint8_t *>(outPos) ) );

        //selection for scalar rest
        uint64_t const * inDataRemainder = inDataCol->get_data_uncompr_start();         
        using t_ve_scalar = scalar<v64<uint64_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)  
        for(size_t i = 0; i < remainderCount; i++){
                *outPosUncompr = t_operator<t_ve_scalar,64>::apply(inDataRemainder[i], val); //write in uncompressed part of the output
                outPosUncompr++; 
                outPos++;
            }
        size_t const outPosCount = outPos - initOutPos;
        size_t const uncomprValuesCnt = outPosCount % blockSize;
        size_t const comprValuesCnt = outPosCount - uncomprValuesCnt; 
        size_t sizeComprByte = comprValuesCnt * sizeof(uint8_t);

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (reinterpret_cast< uint8_t *>(outPosUncompr) - initOutPos) * sizeof(uint8_t), sizeComprByte);
        return outPosCol;        

    }
};

}
#endif 
