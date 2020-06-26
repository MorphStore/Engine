#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_32to32_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_32to32_H

#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/type_packing.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/type_helper.h>
#include <core/operators/interfaces/select.h>

namespace morphstore {

using namespace vectorlib;

template<
        template<class, int> class t_compare,
        class t_vector_extension
>
struct select_t<t_compare, t_vector_extension, type_packing_f<uint32_t >, type_packing_f<uint32_t >> 
    {
    using t_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)

    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    
    static column<type_packing_f<uint32_t >> const * apply(
            column<type_packing_f<uint32_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {
        size_t const inDataCount = inDataCol->get_count_values();
        uint32_t const * inData = inDataCol->get_data();
        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        auto outPosCol = new column<type_packing_f<uint32_t >>(
                bool(outPosCountEstimate)
                // use given estimate
                ? get_size_max_byte_any_len<type_packing_f<uint32_t >>(outPosCountEstimate)
                // use pessimistic estimate
                : get_size_max_byte_any_len<type_packing_f<uint32_t >>(inDataCount)
        );

        uint32_t * outPos = outPosCol->get_data();
        uint32_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint32_t >::m_BlockSize;
        size_t const vectorMultiplier = blockSize / vector_element_count::value;
        size_t vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t remainderCount = inDataCount % blockSize;     
        //std::cout<< "Vectorcount: " << vectorCount << std::endl;
        //std::cout<< "remainderCount: " << remainderCount << std::endl;           
        int startid = 0;      
        vector_t const predicateVector = set1<t_ve, vector_base_t_granularity::value>(val);
        vector_t positionVector = set_sequence<t_ve, vector_base_t_granularity::value>(startid,1);
        vector_t const addVector = set1<t_ve, vector_base_t_granularity::value>(vector_element_count::value);
        for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
            vector_t dataVector = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inData);
            vector_mask_t resultMask =
                t_compare<t_ve,vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            compressstore<t_ve, iov::UNALIGNED, vector_base_t_granularity::value>(outPos, positionVector, resultMask);         
            positionVector = add<t_ve, vector_base_t_granularity::value>::apply(positionVector,addVector);
            outPos += count_matches<t_ve>::apply( resultMask );
            // std::cout<< "outpos address: " << outPos << std::endl;
            // std::cout<< "outpos value: " << *outPos << std::endl;                         
            inData += vector_element_count::value;
        }        
        size_t const vecElCnt = vector_element_count::value;        
        int startidOffsetScalar = vectorCount*vecElCnt;   
        //std::cout<< "startidOffsetScalar: " << startidOffsetScalar << std::endl;
        uint64_t const * inDataRemainder = inDataCol->get_data_uncompr_start(); 
        //selection for scalar rest
        using t_ve_scalar = scalar<v32<uint32_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)     
        for(size_t i = 0; i < remainderCount; i++){
            if(t_compare<t_ve_scalar,32>::apply(inDataRemainder[i], val)) {
                *outPos = i + startidOffsetScalar;
                // std::cout<< "outpos scalar: " << outPos << std::endl;
                // std::cout<< "outpos scalar: " << *outPos << std::endl;
                outPos++;
            }
        }
        size_t const outPosCount = outPos - initOutPos;
        size_t uncomprValuesCnt = outPosCount % blockSize;
        size_t const comprValuesCnt = outPosCount - uncomprValuesCnt; 
        outPos = outPos - uncomprValuesCnt;
        size_t sizeComprByte = comprValuesCnt * sizeof(uint32_t);
        //create padding
        uint64_t * outPosUncompr = reinterpret_cast< uint64_t *>(outPosCol->create_data_uncompr_start(reinterpret_cast< uint8_t *>(outPos) ) );
        //std::cout<< "outPosCount: " << outPosCount << std::endl;
        //std::cout<< "uncomprValuesCnt: " << uncomprValuesCnt << std::endl;
        //std::cout<< "comprValuesCnt: " << comprValuesCnt << std::endl;

        //write in uncompressed part of the output
        for(size_t i = 0; i < uncomprValuesCnt; i++){
            *outPosUncompr = *outPos;
            outPosUncompr++;
            outPos++;

        }
        size_t sizeByte = (reinterpret_cast< uint32_t *>(outPosUncompr) - initOutPos) * sizeof(uint32_t);
        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, sizeByte, sizeComprByte);
        return outPosCol;        
    }
};
}
#endif 

