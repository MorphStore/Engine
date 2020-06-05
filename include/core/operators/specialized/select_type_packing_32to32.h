#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_32to32_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_32to32_H

#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/type_packing.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/typehelper.h>
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
        size_t const vectorCount = inDataCount / vector_element_count::value;
        size_t const remainderCount = inDataCount % vector_element_count::value;
       
        int startid = 0;      
        vector_t const predicateVector = set1<t_ve, vector_base_t_granularity::value>(val);
        vector_t positionVector = set_sequence<t_ve, vector_base_t_granularity::value>(startid,1);
        vector_t const addVector = set1<t_ve, vector_base_t_granularity::value>(vector_element_count::value);
        for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
            vector_t dataVector = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inData);
            //vector_mask_t 
            uint32_t resultMask =
                t_compare<t_ve,vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            compressstore<t_ve, iov::UNALIGNED, vector_base_t_granularity::value>(outPos, positionVector, resultMask);         
            positionVector = add<t_ve, vector_base_t_granularity::value>::apply(positionVector,addVector);
            outPos += count_matches<t_ve>::apply( resultMask );
            inData += vector_element_count::value;
        }        
        int startidOffsetScalar = vectorCount*vector_element_count::value;      
        uint32_t const * inDataRemainder = inDataCol->get_data_uncompr_start(); 
        size_t const vecElCont = vector_element_count::value;

        //selection for scalar rest
        using t_ve_scalar = scalar<v32<uint32_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)     
        for(size_t i = 0; i < remainderCount; i++){
            if(t_compare<t_ve_scalar,32>::apply(inDataRemainder[i], val)) {
                *outPos = i + startidOffsetScalar;
                outPos++;
            }
        }
        size_t const outPosCount = outPos - initOutPos;
        size_t const uncomprValuesCnt = outPosCount % vecElCont;
        size_t const comprValuesCnt = outPosCount - uncomprValuesCnt; 

        outPos = outPos - uncomprValuesCnt;

        size_t sizeComprByte = comprValuesCnt * sizeof(uint32_t);
        //create padding
        uint32_t * outPosUncompr = reinterpret_cast< uint32_t *>(outPosCol->create_data_uncompr_start(reinterpret_cast< uint8_t *>(outPos) ) );

        //write in uncompressed part of the output
        if (comprValuesCnt != 0){
            for(size_t i = 0; i < uncomprValuesCnt; i++){
                *outPosUncompr = *outPos;
                outPosUncompr++;
                outPos++;
            }
        }

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (outPosUncompr - initOutPos) * sizeof(uint32_t), sizeComprByte);
        return outPosCol;        

    }
};
}
#endif 