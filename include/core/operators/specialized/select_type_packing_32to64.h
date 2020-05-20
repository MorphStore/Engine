#ifndef MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_32to64_H
#define MORPHSTORE_CORE_OPERATORS_SPECIALIZED_SELECT_TYPE_PACKING_32to64_H

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
struct select_t//<t_vector_extension, type_packing_f<uint32_t>, type_packing_f<uint32_t>>  
    {
    using t_in_ve = typename TypeHelper<t_vector_extension, uint32_t>::newbasetype;
    using t_out_ve = typename TypeHelper<t_vector_extension, uint64_t>::newbasetype;

    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_in_ve, in_)
    IMPORT_VECTOR_BOILER_PLATE_PREFIX(t_out_ve, out_)

    MSV_CXX_ATTRIBUTE_FORCE_INLINE
    
    static column<type_packing_f<uint64_t >> const * apply(
            column<type_packing_f<uint32_t >> const * const inDataCol,
            in_base_t const val,
            const size_t outPosCountEstimate = 0
    ) {
        size_t const inDataCount = inDataCol->get_count_values();
        in_base_t const * inData = inDataCol->get_data();
        in_base_t const * inDataOriginal = inDataCol->get_data();

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
        size_t const vectorCount = inDataCount / in_vector_element_count::value;
        size_t const remainderCount = inDataCount % in_vector_element_count::value;
       
        int startid = 0;      
        in_vector_t const predicateVector = set1<t_in_ve, in_vector_base_t_granularity::value>(val);
        out_vector_t positionVector = set_sequence<t_out_ve, out_vector_base_t_granularity::value>(startid,1);
        out_vector_t const addVector = set1<t_out_ve, out_vector_base_t_granularity::value>(out_vector_element_count::value);
        for(size_t i = 0; i < vectorCount; ++i) { //selection 
            in_vector_t dataVector = load<t_in_ve, iov::ALIGNED, in_vector_size_bit::value>(inData);
            in_vector_mask_t resultMask = t_compare<t_in_ve,in_vector_base_t_granularity::value>::apply(dataVector, predicateVector);
            out_vector_mask_t resultMaskLow = resultMask & 3; //for the last two values of resultMask
            out_vector_mask_t resultMaskHigh = (resultMask & 12) >> 2; //for value three and four of resultMask
            compressstore<t_out_ve, iov::UNALIGNED, out_vector_size_bit::value>(outPos, positionVector, resultMaskLow);
            positionVector = add<t_out_ve, out_vector_base_t_granularity::value>::apply(positionVector,addVector);       
            compressstore<t_out_ve, iov::UNALIGNED, out_vector_size_bit::value>(outPos, positionVector, resultMaskHigh);
            positionVector = add<t_out_ve, out_vector_base_t_granularity::value>::apply(positionVector,addVector);

            outPos += count_matches<t_in_ve>::apply( resultMask );
            inData += in_vector_element_count::value;

            //std::cout<< "i: " << i << " resultMask: " << resultMask << " resultMaskLow: " << resultMaskLow << " resultMaskHigh: " << resultMaskHigh <<std::endl;
        }        

        //selection for scalar rest 
        // int startidOffsetScalar = vectorCount*vector_element_count::value;       
        // using t_ve_scalar = scalar<v32<uint32_t>>;    
        // IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)            
        // vector_t const predicateVectorScalar = set1<t_ve_scalar, 32>(val);
        // vector_t positionVectorScalar = set_sequence<t_ve_scalar, 32>(startidOffsetScalar,1);
        // vector_t const addVectorScalar = set1<t_ve_scalar, 32>(1);        

        // for(size_t i = 0; i < remainderCount; ++i){ 
        //     vector_t dataVectorScalar = load<t_ve_scalar, iov::ALIGNED, 32>(inData);
        //     vector_mask_t resultMaskScalar =
        //         t_compare<t_ve_scalar,32>::apply(dataVectorScalar, predicateVectorScalar);
        //     compressstore<t_ve_scalar, iov::UNALIGNED, 32>(outPos, positionVectorScalar, resultMaskScalar);
        //     positionVectorScalar = add<t_ve_scalar, 32>::apply(positionVectorScalar,addVectorScalar);

        //     outPos += count_matches<t_ve_scalar>::apply( resultMaskScalar );
        //     inData += 1; 
        // }
       
        int startidOffsetScalar = vectorCount*in_vector_element_count::value;       
        using t_ve_scalar = scalar<v32<uint32_t>>;    
        IMPORT_VECTOR_BOILER_PLATE(t_ve_scalar)            
        for(uint64_t i = startidOffsetScalar; i < startidOffsetScalar+remainderCount; i++){
            if(t_compare<t_ve_scalar,32>::apply(inDataOriginal[i], val)) {
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