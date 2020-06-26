#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_H


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
struct select_t<t_compare, t_vector_extension, type_packing_f<uint64_t >, type_packing_f<uint64_t >>  
    {
    using t_ve = t_vector_extension;
    IMPORT_VECTOR_BOILER_PLATE(t_ve)
    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    
    static column<type_packing_f<uint64_t >> const * apply(
            column<type_packing_f<uint64_t >> const * const inDataCol,
            uint64_t const val,
            const size_t outPosCountEstimate = 0
    ) {        
        size_t const inDataCount = inDataCol->get_count_values();
        uint64_t const * inData = inDataCol->get_data();

        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        auto outPosCol = new column<type_packing_f<uint64_t >>(
                bool(outPosCountEstimate)
                // use given estimate
                ? get_size_max_byte_any_len<type_packing_f<uint64_t >>(outPosCountEstimate)
                // use pessimistic estimate
                : get_size_max_byte_any_len<type_packing_f<uint64_t >>(inDataCount)
        );

        uint64_t * outPos = outPosCol->get_data();
        uint64_t * const initOutPos = outPos; 
        size_t const blockSize = type_packing_f<uint64_t >::m_BlockSize;
        //std::cout<< "blockSize: " << blockSize << std::endl;
        size_t const vectorMultiplier = blockSize / vector_element_count::value;
        size_t const vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        //std::cout<< "Vectorcount: " << vectorCount << std::endl;
        size_t const remainderCount = inDataCount % blockSize;
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
            //std::cout<< "outpos address: " << outPos << std::endl;
            //std::cout<< "outpos value: " << *outPos << std::endl;
            inData += vector_element_count::value;
        }        

        int startidOffsetScalar = vectorCount*vector_element_count::value; 
        //std::cout<< "startidOffsetScalar: " << startidOffsetScalar << std::endl;
        uint64_t const * inDataRemainder = inDataCol->get_data_uncompr_start(); 

        //selection for scalar rest
        using t_ve_scalar = scalar<v64<uint64_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)  
        for(uint64_t i = 0; i < remainderCount; i++){
            if(t_compare<t_ve_scalar,64>::apply(inDataRemainder[i], val)) {
                *outPos = i + startidOffsetScalar;
                outPos++;
            }
        }
        size_t const outPosCount = outPos - initOutPos;
        size_t const uncomprValuesCnt = outPosCount % blockSize;
        size_t const comprValuesCnt = outPosCount - uncomprValuesCnt; 
        outPos = outPos - uncomprValuesCnt;   
        size_t sizeComprByte = comprValuesCnt * sizeof(uint64_t);
        //create padding
        uint64_t * outPosUncompr = reinterpret_cast< uint64_t *>(outPosCol->create_data_uncompr_start(reinterpret_cast< uint8_t *>(outPos) ) );
        //std::cout<< "outPosUncompr: " << outPosUncompr << std::endl;
        //std::cout<< "outPosUncompr value: " << *outPosUncompr << std::endl;

        //write in uncompressed part of the output
        for(size_t i = 0; i < uncomprValuesCnt; i++){
            //std::cout << " in uncompressed part" << std::endl;
            //std::cout<< "outPosUncompr value: " << *outPos << std::endl;               
            *outPosUncompr = *outPos;
            outPosUncompr++;
            outPos++;
        }

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (outPosUncompr - initOutPos) * sizeof(uint64_t), sizeComprByte);
        return outPosCol;        

    }
};

}
#endif 
