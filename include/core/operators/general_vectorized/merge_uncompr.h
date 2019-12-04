/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   merge_uncompr.h
 * Author: Annett
 *
 * Created on 28. Mai 2019, 12:43
 */

#ifndef MERGE_UNCOMPR_H
#define MERGE_UNCOMPR_H

//
// Created by jpietrzyk on 26.04.19.
//

#ifndef MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_MERGE_UNCOMPR_H



#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <cassert>

namespace morphstore {
   
using namespace vectorlib;
   template<class VectorExtension>
   struct merge_sorted_processing_unit {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      struct state_t {
         vector_mask_t m_MaskGreater;//Do we really need this as a state?
         int m_doneLeft;
         int m_doneRight;
         //vector_t m_RotVec;
         state_t(void): m_MaskGreater{ (vector_mask_t)0 } { }
         state_t(vector_mask_t const & p_MaskGreater): m_MaskGreater{ p_MaskGreater } { }
         
         
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
         
            resultMaskEqual         = vectorlib::equal<VectorExtension>::apply(p_Data2Vector, p_Data1Vector);
            p_State.m_MaskGreater   = vectorlib::greater<VectorExtension>::apply(p_Data1Vector, p_Data2Vector);// vec2<vec1?
            
      
         return resultMaskEqual;
      }
   };

   template<class VectorExtension>
   struct merge_sorted_batch {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
    
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static size_t apply(
         base_t  * p_Data1Ptr,//left
         base_t  * p_Data2Ptr,//right
         base_t       * p_OutPtr,
         size_t const    p_CountData1,
         size_t const    p_CountData2,
         typename merge_sorted_processing_unit<VectorExtension>::state_t & p_State
      ) {
          
        //We hope that the larger column has the longest sequential runs, i.e. we get more sequential memory access
        base_t *  pl=p_Data1Ptr;
        base_t *  pr=p_Data2Ptr;
        
        base_t *  endInPosR = p_Data2Ptr+p_CountData2;
        base_t *  endInPosL = p_Data1Ptr+p_CountData1;
        
         //We hope that the larger column has the longest sequential runs, i.e. we get more sequential memory access
        if (p_CountData2 < p_CountData1){
           
            p_Data1Ptr = pr;
            p_Data2Ptr = pl;
            endInPosR = p_Data2Ptr+p_CountData1;
            endInPosL = p_Data1Ptr+p_CountData2;
        }
    
         base_t const * out_init = p_OutPtr;
         
         
        
         vector_t data1Vector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
         
         vector_t data2Vector =vectorlib::load<VectorExtension, vectorlib::iov::ALIGNED, vector_size_bit::value>(p_Data2Ptr);
         
        
         while(p_Data2Ptr < (endInPosR-vector_element_count::value) && p_Data1Ptr < endInPosL){
            
             vector_mask_t resultMaskEqual =
               merge_sorted_processing_unit<VectorExtension>::apply(
                   data1Vector,
                   data2Vector,
                   p_State
               );

           
            
            if((p_State.m_MaskGreater) == 0) { 
                if (resultMaskEqual == 0){
                   *p_OutPtr = vectorlib::extract_value<VectorExtension,vector_base_t_granularity::value>(data1Vector,0);
                    p_OutPtr++;
                }
                p_Data1Ptr ++;
                data1Vector = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(*p_Data1Ptr);
            }else{
                vectorlib::compressstore<VectorExtension,vectorlib::iov::UNALIGNED,vector_base_t_granularity::value>(p_OutPtr,data2Vector,p_State.m_MaskGreater);
                //p_Data2Ptr += __builtin_popcountl(p_State.m_MaskGreater); 
                //p_OutPtr += __builtin_popcountl(p_State.m_MaskGreater);
                p_Data2Ptr += vectorlib::count_matches<VectorExtension>::apply(p_State.m_MaskGreater); 
                p_OutPtr += vectorlib::count_matches<VectorExtension>::apply(p_State.m_MaskGreater);
                data2Vector = vectorlib::load<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_Data2Ptr);
                              
            }
         }
         
         
        //Remainder must be done sequentially
        while(p_Data1Ptr < endInPosL && p_Data2Ptr < endInPosR) {
          
          if(*p_Data1Ptr < *p_Data2Ptr) {
                *p_OutPtr = *p_Data1Ptr;
                p_Data1Ptr++;
            }
            else if(*p_Data2Ptr < *p_Data1Ptr) {
                *p_OutPtr = *p_Data2Ptr;
                p_Data2Ptr++;
            }
            else { // *inPosL == *inPosR
                *p_OutPtr = *p_Data1Ptr;
                p_Data1Ptr++;
                p_Data2Ptr++;
            }
            p_OutPtr++;
        }
         
         
        while (p_Data1Ptr < (endInPosL-vector_element_count::value) ){
            data1Vector=vectorlib::load<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_Data1Ptr);
            vectorlib::store<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_OutPtr,data1Vector);
            p_OutPtr+=vector_element_count::value;
            p_Data1Ptr+=vector_element_count::value;
                   
        }
            
        while (p_Data2Ptr < (endInPosR-vector_element_count::value) ){
            
            data2Vector=vectorlib::load<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_Data2Ptr);
            vectorlib::store<VectorExtension, vectorlib::iov::UNALIGNED, vector_size_bit::value>(p_OutPtr,data2Vector);

            p_Data2Ptr+=vector_element_count::value;
            p_OutPtr+=vector_element_count::value;

        }
        
         //Copy rest, which didn't fit in a vetor register
        while (p_Data1Ptr < endInPosL){
            *p_OutPtr = *p_Data1Ptr;
             p_Data1Ptr++;
             p_OutPtr++;
        }
         
        while (p_Data2Ptr < endInPosR){
             *p_OutPtr = *p_Data2Ptr;
             p_Data2Ptr++;
             p_OutPtr++;
        }
          
        
         return (p_OutPtr-out_init);
      }
   };

   template<class VectorExtension>
   struct merge_sorted_t {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      MSV_CXX_ATTRIBUTE_FORCE_INLINE static
      column<uncompr_f> const *
      apply(
         column< uncompr_f > const * const p_Data1Column,
         column< uncompr_f > const * const p_Data2Column
         
      ) {
         const size_t inData1Count = p_Data1Column->get_count_values();
         const size_t inData2Count = p_Data2Column->get_count_values();
         //assert(inData1Count == p_Data2Column->get_count_values());

         base_t  * inData1Ptr = p_Data1Column->get_data( );
         base_t  * inData2Ptr = p_Data2Column->get_data( );


         size_t const sizeByte =
             p_Data1Column->get_size_used_byte()+p_Data2Column->get_size_used_byte();

         typename merge_sorted_processing_unit<VectorExtension>::state_t vectorState;
         typename merge_sorted_processing_unit<scalar<v64<base_t>>>::state_t scalarState;
         
         auto outDataCol = new column<uncompr_f>(sizeByte);
         base_t * outDataPtr = outDataCol->get_data( );
         
         
         size_t const Count1 = inData1Count;
         size_t const Count2 = inData2Count;
         
         
         int vec_count=merge_sorted_batch<VectorExtension>::apply(inData1Ptr, inData2Ptr, outDataPtr, Count1, Count2,vectorState);
                  
        
         outDataCol->set_meta_data(vec_count, vec_count*sizeof(base_t) );

         return outDataCol;
      }
   };
   
    template<class VectorExtension, class t_out_pos_f, class t_in_pos_l_f, class t_in_pos_r_f>
    column<uncompr_f> const * merge_sorted(
         column< uncompr_f > const * const p_Data1Column,
         column< uncompr_f > const * const p_Data2Column
        
      ){
        return merge_sorted_t<VectorExtension>::apply(p_Data1Column,p_Data2Column);
    }


}


#endif //MORPHSTORE_CORE_OPERATORS_GENERAL_VECTORIZED_INTERSECT_UNCOMPR_H

#endif /* MERGE_UNCOMPR_H */

