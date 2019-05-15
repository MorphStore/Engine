//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_INTERSECT_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_INTERSECT_UNCOMPR_H


#include <vector/general_vector.h>
#include <vector/primitives/io.h>
#include <vector/primitives/create.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/manipulate.h>
#include <core/utils/preprocessor.h>

#include <cassert>

namespace morphstore {
   //@todo Implementation of vectorized intersect is not correct. Fix this first.
/*
   template<class VectorExtension>
   struct intersect_sorted_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_mask_t m_MaskGreater;
         state_t(void): m_MaskGreater{ (vector_mask_t)0 } { }
         state_t(vector_mask_t const & p_MaskGreater): m_MaskGreater{ p_MaskGreater } { }
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_mask_t const &
      apply(
         vector_t const & p_Data1Vector,
         vector_t const & p_Data2Vector,  //@todo: should it be non-const because of rotation?
         state_t        & p_State
      ) {
         vector_mask_t resultMaskEqual    = 0;
         for(size_t i = 0; i < vector_element_count::value; ++i ) {
            resultMaskEqual         |= vector::equal<VectorExtension>::apply(p_Data2Vector, p_Data1Vector);
            p_State.m_MaskGreater   |= vector::greater<VectorExtension>::apply(p_Data2Vector, p_Data1Vector);
            vector::rotate<VectorExtension>(p_Data2Vector);
         }
         return resultMaskEqual;
      }
   };

   template<class VectorExtension>
   struct intersect_sorted_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         bool m_ChangedLeft;
         state_t(void): m_ChangedLeft{ false } { }
         state_t(bool const & p_ChangedLeft): m_ChangedLeft{ p_ChangedLeft } { }
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static void apply(
         base_t const *& p_Data1Ptr,
         base_t const *& p_Data2Ptr,
         base_t       *& p_OutPtr,
         size_t const    p_CountData1,
         size_t const    p_CountData2,
         typename intersect_sorted_processing_unit<VectorExtension>::state_t & p_State
      ) {
         vector_t data1Vector;
         vector_t data2Vector;

         vector_mask_t resultMaskEqual =
            intersect_sorted_processing_unit<VectorExtension>::apply(
                data1Vector,
                data2Vector,
                p_State
            );
         vector::compressstore<VectorExtension, vector::iov::UNALIGNED, vector_base_t_granularity>(
            p_OutPtr,
            data1Vector,
            resultMaskEqual
         );
         p_OutPtr += __builtin_popcount(resultMaskEqual);
         if(p_State.m_MaskGreater == 0) { //@todo: Original resultMaskEqual | p_State.m_MaskGreater. resultMaskEqual is not necessary, is it?
            p_Data2Ptr += vector_element_count::value;
            data2Vector = vector::load<VectorExtension, vector::iov::ALIGNED, vector_base_t_granularity>(
               p_Data2Ptr
            );
            changed_left = false;
         }
      }
   };

   template<class VectorExtension, template<class> class Comparator>
   struct calc_binary {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      column<uncompr_f> const *
      apply(
         column< uncompr_f > const * const p_Data1Column,
         column< uncompr_f > const * const p_Data2Column,
         const size_t p_OutPosCountEstimate = 0
      ) {
         const size_t inData1Count = p_Data1Column->get_count_values();
         assert(inData1Count == p_Data2Column->get_count_values());

         base_t const * inData1Ptr = p_Data1Column->get_data( );
         base_t const * inData2Ptr = p_Data2Column->get_data( );


         size_t const sizeByte =
            bool(p_OutPosCountEstimate)
            ? (p_OutPosCountEstimate * sizeof(base_t))
            : p_Data1Column->get_size_used_byte();

         auto outDataCol = new column<uncompr_f>(sizeByte);
         base_t * outDataPtr = outDataCol->get_data( );

         size_t const vectorCount = inData1Count / vector_element_count::value;
         size_t const remainderCount = inData1Count % vector_element_count::value;

         calc_binary_batch<VectorExtension,Comparator>::apply(inData1Ptr, inData2Ptr, outDataPtr, vectorCount);
         calc_binary_batch<vector::scalar<base_t>,Comparator>::apply(inData1Ptr, inData2Ptr, outDataPtr, remainderCount);

         outDataCol->set_meta_data(inData1Count, sizeByte);

         return outDataCol;
      }
   };
*/

}


#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_INTERSECT_UNCOMPR_H