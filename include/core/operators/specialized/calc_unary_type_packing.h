#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_CALC_UNARY_TYPE_PACKING_H

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
        class t_vector_extension
>
struct calc_unary_t<t_operator, t_vector_extension, type_packing_f<uint64_t >, type_packing_f<uint64_t >>  
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
        size_t const vectorMultiplier = blockSize / vector_element_count::value;
        size_t const vectorCount = (inDataCount / blockSize) * vectorMultiplier;
        size_t const remainderCount = inDataCount % blockSize;
        // std::cout<< "Vectorcount: " << vectorCount << std::endl;
        // std::cout<< "remainderCount: " << remainderCount << std::endl;
        // std::cout<< "blockSize: " << blockSize << std::endl;


        vector_t const predicateVector = set1<t_ve, vector_base_t_granularity::value>(val);
        for(size_t i = 0; i < vectorCount; ++i) { //selection for compressed part
            vector_t dataVector = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inData);
            vector_t result =
            	t_operator<t_ve,vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            vectorlib::store<t_ve, iov::ALIGNED, vector_size_bit::value>(outPos, result);
            outPos += vector_element_count::value;
            inData += vector_element_count::value;
        }        

        //create padding
        uint64_t * outPosUncompr = reinterpret_cast< uint64_t *>(outPosCol->create_data_uncompr_start(reinterpret_cast< uint8_t *>(outPos) ) );

        //selection for scalar rest
        uint64_t const * inDataRemainder = inDataCol->get_data_uncompr_start();         
        using t_ve_scalar = scalar<v64<uint64_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)  
        for(size_t i = 0; i < remainderCount; i++){
        	//std::cout<< "in scalar" << std::endl;
                *outPosUncompr = t_operator<t_ve_scalar,64>::apply(inDataRemainder[i], val); //write in uncompressed part of the output
                outPosUncompr++; 
                outPos++;
            }
        size_t const outPosCount = outPos - initOutPos;
        size_t const uncomprValuesCnt = outPosCount % blockSize;
        size_t const comprValuesCnt = outPosCount - uncomprValuesCnt; 
        size_t sizeComprByte = comprValuesCnt * sizeof(uint64_t);

        // #log, sizeByte , sizeComprByte
        outPosCol->set_meta_data(uncomprValuesCnt + comprValuesCnt, (outPosUncompr - initOutPos) * sizeof(uint64_t), sizeComprByte);
        return outPosCol;        

    }
};

}
#endif 
