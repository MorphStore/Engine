#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_16to32_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_16to32_H

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
struct calc_unary_t<t_operator, vectorlib::sse<vectorlib::v128<uint64_t>>, type_packing_f<uint32_t >, type_packing_f<uint16_t >>  
    {
    using t_vector_extension = vectorlib::sse<vectorlib::v128<uint64_t>>;
    using t_in_ve = typename TypeHelper<t_vector_extension, uint16_t>::newbasetype;
    using t_out_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype; 
    
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_in_ve, in_)
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_out_ve, out_)
    MSV_CXX_ATTRIBUTE_FORCE_INLINE

                                    
    static column<type_packing_f<uint32_t >> const * apply(
            column<type_packing_f<uint16_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {        
        size_t const inDataCount = inDataCol->get_count_values();
        in_base_t const * inData = inDataCol->get_data();

        auto outPosCol = new column<type_packing_f<uint32_t >>( 
                bool(outPosCountEstimate)
                // use given estimate                          
                ? get_size_max_byte_any_len<type_packing_f<uint32_t >>(outPosCountEstimate)
                // use pessimistic estimate                    
                : get_size_max_byte_any_len<type_packing_f<uint32_t >>(inDataCount)
        );

        out_base_t * outPos = outPosCol->get_data();
        out_base_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint64_t >::m_BlockSize;
        size_t const vectorMultiplier = blockSize / in_vector_element_count::value;
        size_t const vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t const remainderCount = inDataCount % blockSize;

        out_vector_t const predicateVector = set1<t_out_ve, out_vector_base_t_granularity::value>(val);
        for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
                //16 bit expanded to 32 bit
                // out_vector_t dataVectorExpanded = _mm_cvtepu16_epi32(dataVector);
                // out_vector_t result =
                //  t_operator<t_out_ve, out_vector_base_t_granularity::value>::apply(dataVectorExpanded, predicateVector);
                // vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, result);
                // outPos += in_vector_element_count::value;
                // inData += in_vector_element_count::value;
                // std::cout<< "in compressed "  << std::endl;
            in_vector_t dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData); //load 8 values for sse
         // uint16_t number1 = 0;
         // for(int i=0; i < 8; i++){
         //     number1 = extract_value<t_in_ve, in_vector_base_t_granularity::value>(dataVector, i);
         //     std::cout << "dataVector: " << number1 << std::endl;
         // }
            out_vector_t dataVectorExpanded = _mm_shuffle_epi8(dataVector, _mm_set_epi8(127,127,7,6,127,127,5,4,
               127,127,3,2,127,127,1,0));
            out_vector_t result =
                t_operator<t_out_ve, out_vector_base_t_granularity::value>::apply(dataVectorExpanded, predicateVector);
            vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, result);
            outPos += out_vector_element_count::value;

            out_vector_t dataVectorExpanded2 = _mm_shuffle_epi8(dataVector, _mm_set_epi8(127,127,15,14,127,127,13,12,
               127,127,11,10,127,127,9,8));
            out_vector_t result2 =
                t_operator<t_out_ve, out_vector_base_t_granularity::value>::apply(dataVectorExpanded2, predicateVector);
            vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, result2);
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
        size_t sizeComprByte = comprValuesCnt * sizeof(uint32_t);

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (reinterpret_cast< uint32_t *>(outPosUncompr) - initOutPos) * sizeof(uint32_t), sizeComprByte);
        return outPosCol;        

    }
};



template<
        template<class, int> class t_operator
>                                                                    
struct calc_unary_t<t_operator, vectorlib::avx2<vectorlib::v256<uint64_t> >, type_packing_f<uint32_t >, type_packing_f<uint16_t >>  
    {
    using t_vector_extension = vectorlib::avx2<vectorlib::v256<uint64_t> >;
    using t_in_ve = typename TypeHelper<t_vector_extension, uint16_t>::newbasetype;
    using t_out_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype; 
    
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_in_ve, in_)
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_out_ve, out_)
    MSV_CXX_ATTRIBUTE_FORCE_INLINE

                                    
    static column<type_packing_f<uint32_t >> const * apply(
            column<type_packing_f<uint16_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {        
        size_t const inDataCount = inDataCol->get_count_values();
        in_base_t const * inData = inDataCol->get_data();

        auto outPosCol = new column<type_packing_f<uint32_t >>( 
                bool(outPosCountEstimate)
                // use given estimate                          
                ? get_size_max_byte_any_len<type_packing_f<uint32_t >>(outPosCountEstimate)
                // use pessimistic estimate                    
                : get_size_max_byte_any_len<type_packing_f<uint32_t >>(inDataCount)
        );

        out_base_t * outPos = outPosCol->get_data();
        out_base_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint64_t >::m_BlockSize;
        size_t const vectorMultiplier = blockSize / in_vector_element_count::value;
        size_t const vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t const remainderCount = inDataCount % blockSize;

        out_vector_t const predicateVector = set1<t_out_ve, out_vector_base_t_granularity::value>(val);
        for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
            in_vector_t dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData); //load 16 values for avx2

            out_vector_t dataVectorPermuted = _mm256_permutevar8x32_epi32(dataVector, _mm256_set_epi32(7,6,3,2,3,2,1,0));
            out_vector_t dataVectorExpanded = _mm256_shuffle_epi8(dataVectorPermuted, _mm256_set_epi8(127,127,23,22,127,127,21,20,127,127,19,18,
                127,127,17,16,127,127,7,6,127,127,5,4,127,127,3,2,127,127,1,0));
            out_vector_t result =
                t_operator<t_out_ve, out_vector_base_t_granularity::value>::apply(dataVectorExpanded, predicateVector);
            vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, result);
            outPos += out_vector_element_count::value;

            out_vector_t dataVectorPermuted2 = _mm256_permutevar8x32_epi32(dataVector, _mm256_set_epi32(7,6,7,6,3,2,5,4));            
            out_vector_t dataVectorExpanded2 = _mm256_shuffle_epi8(dataVectorPermuted2, _mm256_set_epi8(127,127,23,22,127,127,21,20,127,127,19,18,
                127,127,17,16,127,127,7,6,127,127,5,4,127,127,3,2,127,127,1,0));
            out_vector_t result2 =
                t_operator<t_out_ve, out_vector_base_t_granularity::value>::apply(dataVectorExpanded2, predicateVector);
            vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, result2);
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
        size_t sizeComprByte = comprValuesCnt * sizeof(uint32_t);

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (reinterpret_cast< uint32_t *>(outPosUncompr) - initOutPos) * sizeof(uint32_t), sizeComprByte);
        return outPosCol;        

    }
};


}
#endif 
