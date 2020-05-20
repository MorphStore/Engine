#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_H


#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/type_packing.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/typehelper.h>


namespace morphstore {

using namespace vectorlib;

template<
        template<class, int> class t_compare,
        class t_vector_extension
>
struct select_t//<t_vector_extension, type_packing_f<uint64_t >, type_packing_f<uint64_t >>  
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
        uint64_t const * inDataOriginal = inDataCol->get_data();

        // If no estimate is provided: Pessimistic allocation size (for
        // uncompressed data), reached only if all input data elements pass the
        // selection.
        auto outPosCol = new column<type_packing_f<uint64_t >>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : inDataCol->get_size_used_byte()
        );

        uint64_t * outPos = outPosCol->get_data();
        uint64_t * const initOutPos = outPos; 
        size_t const vectorCount = inDataCount / vector_element_count::value;
        size_t const remainderCount = inDataCount % vector_element_count::value;
       
        int startid = 0;      
        vector_t const predicateVector = set1<t_ve, vector_base_t_granularity::value>(val);
        vector_t positionVector = set_sequence<t_ve, vector_base_t_granularity::value>(startid,1);
        vector_t const addVector = set1<t_ve, vector_base_t_granularity::value>(vector_element_count::value);
        for(size_t i = 0; i < vectorCount; ++i) { //selection 
            vector_t dataVector = load<t_ve, iov::ALIGNED, vector_size_bit::value>(inData);
            vector_mask_t resultMask =
                t_compare<t_ve,vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            compressstore<t_ve, iov::UNALIGNED, vector_size_bit::value>(outPos, positionVector, resultMask);
            positionVector = add<t_ve, vector_base_t_granularity::value>::apply(positionVector,addVector);

            outPos += count_matches<t_ve>::apply( resultMask );
            inData += vector_element_count::value;
        }        

        
        //selection for scalar rest
        // int startidOffsetScalar = vectorCount*vector_element_count::value;       
        // using t_ve_scalar = scalar<v64<uint64_t>>;    
        // IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)            
        // vector_t const predicateVectorScalar = set1<t_ve_scalar, 64>(val);
        // vector_t positionVectorScalar = set_sequence<t_ve_scalar, 64>(startidOffsetScalar,1);
        // vector_t const addVectorScalar = set1<t_ve_scalar, 64>(1);        

        // for(size_t i = 0; i < remainderCount; ++i){ 
        //     vector_t dataVectorScalar = load<t_ve_scalar, iov::ALIGNED, 64>(inData);
        //     vector_mask_t resultMaskScalar =
        //         t_compare<t_ve_scalar,64>::apply(dataVectorScalar, predicateVectorScalar);
        //     compressstore<t_ve_scalar, iov::UNALIGNED, 64>(outPos, positionVectorScalar, resultMaskScalar);
        //     positionVectorScalar = add<t_ve_scalar, 64>::apply(positionVectorScalar,addVectorScalar);

        //     outPos += count_matches<t_ve_scalar>::apply( resultMaskScalar );
        //     inData += 1; 
        // }
       
        int startidOffsetScalar = vectorCount*vector_element_count::value;       
        using t_ve_scalar = scalar<v64<uint64_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)            
        for(uint64_t i = startidOffsetScalar; i < startidOffsetScalar+remainderCount; i++){
            //std::cout<< "startidOffsetScalar" << startidOffsetScalar << std::endl;
            //std::cout<< "inDataOriginal[i]" << inDataOriginal[i] << std::endl;
            if(t_compare<t_ve_scalar,64>::apply(inDataOriginal[i], val)) {
                *outPos = i;
                outPos++;
            }
        }

        size_t const outPosCount = outPos - initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));

        return outPosCol;        

    }
};


}
#endif 