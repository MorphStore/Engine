#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_EXPAND_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_EXPAND_H

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
        template<class, int> class t_operator,
        class t_vector_extension,
        class t_out_pos_f,
        class t_in_data_f        
>                                                                    
//struct calc_unary_t<t_operator, t_vector_extension, type_packing_f<uint32_t >, type_packing_f<uint16_t >>  
struct calc_unary_t<t_operator, t_vector_extension, type_packing_f<t_out_pos_f >, type_packing_f<t_in_data_f >>  
    {
    using t_ve = t_vector_extension;
    // using t_in_ve = typename TypeHelper<t_vector_extension, uint16_t>::newbasetype;
    // using t_out_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype; 
    using t_in_ve = typename TypeHelper<t_vector_extension, t_in_data_f>::newbasetype;
    using t_out_ve = typename TypeHelper<t_vector_extension, t_out_pos_f>::newbasetype; 
    
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_in_ve, in_)
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_out_ve, out_)
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
    MSV_CXX_ATTRIBUTE_FORCE_INLINE

                                    
    static column<type_packing_f<out_base_t >> const * apply(
            column<type_packing_f<in_base_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {        
        size_t const inDataCount = inDataCol->get_count_values();
        in_base_t const * inData = inDataCol->get_data();

        auto outPosCol = new column<type_packing_f<out_base_t >>( 
                bool(outPosCountEstimate)
                // use given estimate                          
                ? get_size_max_byte_any_len<type_packing_f<out_base_t >>(outPosCountEstimate)
                // use pessimistic estimate                    
                : get_size_max_byte_any_len<type_packing_f<out_base_t >>(inDataCount)
        );

        out_base_t * outPos = outPosCol->get_data();
        out_base_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint64_t >::m_BlockSize;

        size_t vectorMultiplier;
        if(out_vector_base_t_granularity::value > in_vector_base_t_granularity::value){
            vectorMultiplier = blockSize / in_vector_element_count::value;
        } 
        else{
            vectorMultiplier = blockSize / out_vector_element_count::value;
        }
        size_t const vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t const remainderCount = inDataCount % blockSize;

        if(out_vector_base_t_granularity::value > in_vector_base_t_granularity::value){ //if it expands
            out_vector_t const predicateVector = set1<t_out_ve, out_vector_base_t_granularity::value>(val);
            for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
                in_vector_t dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData);           
                out_vector_t dataVectorExpanded;
                out_vector_t result;
                for(size_t j=1; j <= out_vector_base_t_granularity::value / in_vector_base_t_granularity::value; j++){
                    dataVectorExpanded = expandOrCompact<t_ve, in_vector_base_t_granularity::value, out_vector_base_t_granularity::value>::apply(dataVector, j);
                    result = t_operator<t_out_ve, out_vector_base_t_granularity::value>::apply(dataVectorExpanded, predicateVector);
                    vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, result);
                    outPos += out_vector_element_count::value;
                }
                inData += in_vector_element_count::value;
            }        
        }
        else if(in_vector_base_t_granularity::value > out_vector_base_t_granularity::value){ //if it compacts
            in_vector_t const predicateVector = set1<t_in_ve, in_vector_base_t_granularity::value>(val);
            for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
                //first add/substract datavector and predicateVector (32 bit) and then compress the result to 8 bit
                //create multiple resultVectorCompressed vectors and combine them to one via vectorlib::bitwise_or
                //store this one final vector
                out_vector_t resultVectorCompressed = set1<t_out_ve, out_vector_base_t_granularity::value>(0);
                in_vector_t dataVector;
                in_vector_t result;
                out_vector_t temp;
                for(size_t j=1; j<= in_vector_base_t_granularity::value / out_vector_base_t_granularity::value; j++){
                    dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData);;
                    result = t_operator<t_in_ve, in_vector_base_t_granularity::value>::apply(dataVector, predicateVector);
                    temp = expandOrCompact<t_ve, in_vector_base_t_granularity::value, out_vector_base_t_granularity::value>::apply(result, j);
                    inData += in_vector_element_count::value;
                    resultVectorCompressed = bitwise_or<t_out_ve>(resultVectorCompressed, temp);
                }
                vectorlib::store<t_out_ve, iov::ALIGNED, out_vector_size_bit::value>(outPos, resultVectorCompressed);
                outPos += out_vector_element_count::value;
            }                
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
        size_t sizeComprByte = comprValuesCnt * sizeof(out_base_t);

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (reinterpret_cast< out_base_t *>(outPosUncompr) - initOutPos) * sizeof(out_base_t), sizeComprByte);
        return outPosCol;        

    }
};

}
#endif 