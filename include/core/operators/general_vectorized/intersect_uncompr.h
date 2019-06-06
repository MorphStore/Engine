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
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/manipulate_scalar.h>

#include <cassert>

namespace morphstore {
   
using namespace vector;
   template<class VectorExtension>
   struct intersect_sorted_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_mask_t m_MaskLess;//Do we really need this as a state?
         int m_doneLeft;
         int m_doneRight;
         //vector_t m_RotVec;
         state_t(void): m_MaskLess{ (vector_mask_t)0 } { }
         state_t(vector_mask_t const & p_MaskLess): m_MaskLess{ p_MaskLess } { }
         
         
      };
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static
      vector_mask_t
      apply(
         vector_t const p_Data1Vector,
         vector_t const p_Data2Vector,
              
         state_t    &    p_State
      ) {
         vector_mask_t resultMaskEqual    = 0;
         
            resultMaskEqual      = vector::equal<VectorExtension>::apply(p_Data2Vector, p_Data1Vector);
            p_State.m_MaskLess   = vector::less<VectorExtension>::apply(p_Data2Vector, p_Data1Vector);// vec2<vec1?
            
      
         return resultMaskEqual;
      }
   };

   template<class VectorExtension>
   struct intersect_sorted_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
    
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t apply(
         base_t  const * p_Data1Ptr,//left
         base_t  const * p_Data2Ptr,//right
         base_t       * p_OutPtr,
         size_t const    p_CountData1,
         size_t const    p_CountData2,
         typename intersect_sorted_processing_unit<VectorExtension>::state_t & p_State
      ) {
          
         base_t const * Data1Ptr_start = p_Data1Ptr;
         base_t const * Data2Ptr_start = p_Data2Ptr;
         base_t const * out_init = p_OutPtr;
         
         
         const base_t * const endInPosR = p_Data2Ptr+p_CountData2;
         const base_t * const endInPosL = p_Data1Ptr+p_CountData1;
         vector_t data1Vector = vector::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
         
         vector_t data2Vector =vector::load<VectorExtension, vector::iov::ALIGNED, vector_size_bit::value>(p_Data2Ptr);
         
         vector_mask_t full_hit = vector::equal<VectorExtension>::apply(data1Vector, data1Vector);
         
         while(p_Data2Ptr < endInPosR && p_Data1Ptr < endInPosL){
            
             vector_mask_t resultMaskEqual =
               intersect_sorted_processing_unit<VectorExtension>::apply(
                   data1Vector,
                   data2Vector,
                   p_State
               );

             if (resultMaskEqual!=0) {
                *p_OutPtr=*p_Data1Ptr;//if keys are not unique, use a compressstore here
      
                p_OutPtr++;
             }
            
            if((p_State.m_MaskLess) == 0) { //@todo: Original resultMaskEqual | p_State.m_MaskGreater. resultMaskEqual is not necessary, is it?
               p_Data1Ptr++;
               data1Vector = vector::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
               
            }else{
                if((p_State.m_MaskLess) == full_hit) { 
                    p_Data2Ptr += vector_element_count::value;
                    data2Vector = vector::load<VectorExtension, vector::iov::UNALIGNED, vector_size_bit::value>(
                       p_Data2Ptr
                    );
                    
                }else{
                    p_Data1Ptr++;
                    data1Vector = vector::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
                    p_Data2Ptr += __builtin_popcount(p_State.m_MaskLess);
                    data2Vector = vector::load<VectorExtension, vector::iov::UNALIGNED, vector_size_bit::value>(
                       p_Data2Ptr
                    );
                }
               
            }
         }
         
         p_State.m_doneLeft = p_Data1Ptr-Data1Ptr_start;
         p_State.m_doneRight = p_Data2Ptr-Data2Ptr_start;
         
        // if ((p_State.m_MaskLess) != 0) p_State.m_doneRight-=__builtin_popcount(p_State.m_MaskLess);
        // if ((p_State.m_MaskLess) != full_hit) p_State.m_doneLeft--;
         return (p_OutPtr-out_init);
      }
   };

   template<class VectorExtension>
   struct intersect_sorted_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      column<uncompr_f> const *
      apply(
         column< uncompr_f > const * const p_Data1Column,
         column< uncompr_f > const * const p_Data2Column,
         const size_t p_OutPosCountEstimate = 0
      ) {
         const size_t inData1Count = p_Data1Column->get_count_values();
         const size_t inData2Count = p_Data2Column->get_count_values();
         //assert(inData1Count == p_Data2Column->get_count_values());

         base_t  const * inData1Ptr = p_Data1Column->get_data( );
         base_t  const * inData2Ptr = p_Data2Column->get_data( );

         size_t const sizeByte =
            bool(p_OutPosCountEstimate)
            ? (p_OutPosCountEstimate * sizeof(base_t))
            : std::min(
                    p_Data1Column->get_size_used_byte(),
                    p_Data2Column->get_size_used_byte()
            );

         typename intersect_sorted_processing_unit<VectorExtension>::state_t vectorState;
         typename intersect_sorted_processing_unit<scalar<v64<uint64_t>>>::state_t scalarState;
         
         auto outDataCol = new column<uncompr_f>(sizeByte);
         base_t * outDataPtr = outDataCol->get_data( );
         
         
         size_t const Count1 = inData1Count;
         

         size_t const Count2 = inData2Count-vector_element_count::value;
         
         int vec_count=0;
         vec_count=intersect_sorted_batch<VectorExtension>::apply(inData1Ptr, inData2Ptr, outDataPtr, Count1, Count2,vectorState);
                  
         int scalar_count=0;

         scalar_count=intersect_sorted_batch<scalar<v64<uint64_t>>>::apply(inData1Ptr+vectorState.m_doneLeft, inData2Ptr+vectorState.m_doneRight, outDataPtr+vec_count,inData1Count-vectorState.m_doneLeft, inData2Count-vectorState.m_doneRight,scalarState);

         outDataCol->set_meta_data((vec_count+scalar_count), (vec_count+scalar_count)*sizeof(base_t) );

         return outDataCol;
      }
   };
   
    template<class VectorExtension>
    column<uncompr_f> const * intersect_sorted(
         column< uncompr_f > const * const p_Data1Column,
         column< uncompr_f > const * const p_Data2Column,
         const size_t p_OutPosCountEstimate = 0
      ) {
        
        return intersect_sorted_t<VectorExtension>::apply(p_Data1Column,p_Data2Column,p_OutPosCountEstimate);
    }

}


#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_INTERSECT_UNCOMPR_H