//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_UNCOMPR_H

#include <vector/general_vector.h>
#include <vector/primitives/io.h>
#include <core/utils/preprocessor.h>

#include <vector/scalar/extension_skalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>

namespace morphstore {

       using namespace vector;
       
   template<class VectorExtension>
   struct project_t_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static vector_t const & apply(
         base_t const * const p_DataPtr,
         vector_t const & p_PosVector
      ) {
         //@todo: Is it better to avoid gather here?
         return (vector::gather<VectorExtension, vector::iov::UNALIGNED, vector_size_bit::value>(p_DataPtr, p_PosVector));
      }
   };

   template<class VectorExtension>
   struct project_t_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t const * const p_DataPtr,
         base_t const *& p_PosPtr,
         base_t *& p_OutPtr,
         size_t const p_Count
      ) {
         for(size_t i = 0; i < p_Count; ++i) {
            vector_t posVector = vector::load<VectorExtension, vector::iov::ALIGNED, vector_size_bit::value>(p_PosPtr);
            vector_t lookupVector =
               project_t_processing_unit<VectorExtension>::apply(
                  p_DataPtr,
                  posVector
               );
            vector::store<VectorExtension, vector::iov::ALIGNED, vector_size_bit::value>(p_OutPtr, lookupVector);
            p_PosPtr += vector_element_count::value;
            p_OutPtr += vector_element_count::value;
         }
      }
   };

   template<class VectorExtension>
   struct project_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      column<uncompr_f> const *
      apply(
         column< uncompr_f > const * const p_DataColumn,
         column< uncompr_f > const * const p_PosColumn
      ) {
         size_t const inPosCount = p_PosColumn->get_count_values();
         size_t const inUsedBytes = p_PosColumn->get_size_used_byte();
         size_t const vectorCount = inPosCount / vector_element_count::value;
         size_t const remainderCount = inPosCount % vector_element_count::value;
         base_t const * const inDataPtr = p_DataColumn->get_data( );
         base_t const * inPosPtr = p_PosColumn->get_data( );
         auto outDataCol = new column<uncompr_f>(inUsedBytes);
         base_t * outDataPtr = outDataCol->get_data( );
         project_t_batch<VectorExtension>::apply(inDataPtr, inPosPtr, outDataPtr, vectorCount);
         project_t_batch<scalar<v64<uint64_t>>>::apply(inDataPtr, inPosPtr, outDataPtr, remainderCount);

         outDataCol->set_meta_data(inPosCount, inUsedBytes);

         return outDataCol;
      }
   };


}



#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_UNCOMPR_H

