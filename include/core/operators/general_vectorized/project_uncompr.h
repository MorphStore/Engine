//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_UNCOMPR_H

#include <core/utils/preprocessor.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {

       using namespace vectorlib;
       
   template<class VectorExtension>
   struct project_t_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static vector_t apply(
         base_t const * p_DataPtr,
         vector_t p_PosVector
      ) {
         //@todo: Is it better to avoid gather here?
         return (vectorlib::gather<VectorExtension, vector_size_bit::value, sizeof(base_t)>(p_DataPtr, p_PosVector));
      }
   };

   template<class VectorExtension>
   struct project_t_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t const *& p_DataPtr,
         base_t const *& p_PosPtr,
         base_t *& p_OutPtr,
         size_t const p_Count
      ) {
         for(size_t i = 0; i < p_Count; ++i) {
            vector_t posVector = vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_PosPtr);
            vector_t lookupVector =
               project_t_processing_unit<VectorExtension>::apply(
                  p_DataPtr,
                  posVector
               );
            vectorlib::store<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_OutPtr, lookupVector);
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
         base_t const * inDataPtr = p_DataColumn->get_data( );
         base_t const * inPosPtr = p_PosColumn->get_data( );
         auto outDataCol = new column<uncompr_f>(inUsedBytes);
         base_t * outDataPtr = outDataCol->get_data( );
         project_t_batch<VectorExtension>::apply(inDataPtr, inPosPtr, outDataPtr, vectorCount);
         project_t_batch<scalar<v64<base_t>>>::apply(inDataPtr, inPosPtr, outDataPtr, remainderCount);

         outDataCol->set_meta_data(inPosCount, inUsedBytes);

         return outDataCol;
      }
   };

    template<class t_vector_extension, class t_out_data_f, class t_in_data_f, class t_in_pos_f>
    column<uncompr_f> const * project(
         column< uncompr_f > const * const p_Data1Column,
         column< uncompr_f > const * const p_Data2Column
      ){
        return project_t<t_vector_extension>::apply(p_Data1Column,p_Data2Column);
    }

}



#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_PROJECT_UNCOMPR_H

