#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_64to32_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_64to32_H

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
struct select_t<t_compare, t_vector_extension, type_packing_f<uint32_t >, type_packing_f<uint64_t >>  
    {

    using t_in_ve = typename TypeHelper<t_vector_extension, uint64_t>::newbasetype;
    using t_out_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype;
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_in_ve, in_)
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_out_ve, out_)    
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    
    static column<type_packing_f<uint32_t >> const * apply(
            column<type_packing_f<uint64_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {
        size_t const inDataCount = inDataCol->get_count_values();
        uint64_t const * inData = inDataCol->get_data();

        auto outPosCol = new column<type_packing_f<uint32_t >>(
                bool(outPosCountEstimate)
                // use given estimate
                ? get_size_max_byte_any_len<type_packing_f<uint32_t >>(outPosCountEstimate)
                // use pessimistic estimate
                : get_size_max_byte_any_len<type_packing_f<uint32_t >>(inDataCount)
        );
        uint32_t * outPos = outPosCol->get_data();
        uint32_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint64_t >::m_BlockSize;
        size_t const vectorMultiplier = blockSize / out_vector_element_count::value;
        size_t vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t remainderCount = inDataCount % blockSize; 
        //std::cout<< "Vectorcount: " << vectorCount << std::endl;
        //std::cout<< "remainderCount: " << remainderCount << std::endl;
        int startid = 0;      
        in_vector_t const predicateVector = set1<t_in_ve, in_vector_base_t_granularity::value>(val);
        out_vector_t positionVector = set_sequence<t_out_ve, out_vector_base_t_granularity::value>(startid,1);
        out_vector_t const addVector = set1<t_out_ve, out_vector_base_t_granularity::value>(out_vector_element_count::value);
        for(size_t i = 0; i < vectorCount; ++i) { //selection 
            //load and mask two times
            in_vector_t dataVector1 = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData);
            in_vector_mask_t resultMask1 =
                t_compare<t_in_ve,in_vector_base_t_granularity::value>::apply(dataVector1, predicateVector);
            inData += in_vector_element_count::value; 
            in_vector_t dataVector2 = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData);
            in_vector_mask_t resultMask2 =
                t_compare<t_in_ve,in_vector_base_t_granularity::value>::apply(dataVector2, predicateVector);
            inData += in_vector_element_count::value; 
            //combine two masks to one mask, then compressstore of the one mask
            out_vector_mask_t resultMask = (resultMask2 << in_vector_element_count::value ) | (((1 << in_vector_element_count::value) - 1) & resultMask1);
            compressstore<t_out_ve, iov::UNALIGNED, out_vector_base_t_granularity::value>(outPos, positionVector, resultMask);
            //std::cout<< "outpos address: " << outPos << std::endl;
            //std::cout<< "outpos value: " << *outPos << std::endl;            
            positionVector = add<t_out_ve, out_vector_base_t_granularity::value>::apply(positionVector,addVector);
            outPos += count_matches<t_out_ve>::apply( resultMask );
        }        
        size_t const vecElCnt = out_vector_element_count::value;
        int startidOffsetScalar = vectorCount*vecElCnt;      
        uint64_t const * inDataRemainder = inDataCol->get_data_uncompr_start(); 
        //std::cout<< "startidOffsetScalar: " << startidOffsetScalar << std::endl;

        //selection for scalar rest
        using t_ve_scalar = scalar<v64<uint64_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)     
        for(size_t i = 0; i < remainderCount; i++){
            if(t_compare<t_ve_scalar,64>::apply(inDataRemainder[i], val)) {
                *outPos = i + startidOffsetScalar;
                //std::cout<< "outpos scalar: " << outPos << std::endl;
                //std::cout<< "outpos scalar: " << *outPos << std::endl;                
                outPos++;
            }
        }

        size_t const outPosCount = outPos - initOutPos;
        size_t uncomprValuesCnt = outPosCount % blockSize;      
        size_t const comprValuesCnt = outPosCount - uncomprValuesCnt; 
        outPos = outPos - uncomprValuesCnt;
        //std::cout<< "outPosCount: " << outPosCount << std::endl;
        //std::cout<< "uncomprValuesCnt: " << uncomprValuesCnt << std::endl;
        //std::cout<< "comprValuesCnt: " << comprValuesCnt << std::endl;

        size_t sizeComprByte = comprValuesCnt * sizeof(uint32_t);
        //create padding
        uint64_t * outPosUncompr = reinterpret_cast< uint64_t *>(outPosCol->create_data_uncompr_start(reinterpret_cast< uint8_t *>(outPos) ) );
        //write in uncompressed part of the output
        for(size_t i = 0; i < uncomprValuesCnt; i++){
            //std::cout << " in uncompressed part" << std::endl;
            //std::cout<< "outPosUncompr value: " << *outPos << std::endl;                 
            *outPosUncompr = *outPos;
            outPosUncompr++;
            outPos++;
        }

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (reinterpret_cast< uint32_t *>(outPosUncompr) - initOutPos) * sizeof(uint32_t), sizeComprByte);
        //outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (outPosUncompr - initOutPos) * sizeof(uint32_t), sizeComprByte);
        return outPosCol;         

    }
};


}
#endif 