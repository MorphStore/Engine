/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   intersect_uncompr.h
 * Author: Annett
 *
 * Created on 29. März 2019, 16:29
 */

#ifndef INTERSECT_UNCOMPR_H
#define INTERSECT_UNCOMPR_H


#include <core/operators/interfaces/intersect.h>
#include <core/operators/vectorized/select_uncompr.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/simd/avx2/extension_avx2.h>

#include <immintrin.h>
#include <cstdint>

namespace morphstore {

template<>
const column<uncompr_f> *
intersect_sorted<vectorlib::avx2<vectorlib::v256<uint64_t>>>(
        const column<uncompr_f> * const p_Data1,
        const column<uncompr_f> * const p_Data2,
        const size_t outPosCountEstimate
) {
    const uint64_t * p_Data1Ptr = p_Data1->get_data();
    const uint64_t * p_Data2Ptr = p_Data2->get_data();
    const uint64_t * const endInPosL = (uint64_t*)p_Data1Ptr + p_Data1->get_count_values();
    const uint64_t * const endInPosR = (uint64_t*)p_Data2Ptr + p_Data2->get_count_values();
    
  
         
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if all positions in the smaller input
    // column are contained in the larger input column as well.
    auto outPosCol = new column<uncompr_f>(
            bool(outPosCountEstimate)
            // use given estimate
            ? (outPosCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : std::min(
                    p_Data1->get_size_used_byte(),
                    p_Data2->get_size_used_byte()
            )
    );
    uint64_t * p_OutPtr = outPosCol->get_data();
    uint64_t const * out_init = p_OutPtr;
    

    __m256i data1Vector;
    __m256i data2Vector;
    int mask=0;
    int mask_greater_than=0;
    data1Vector=_mm256_loadu_si256((__m256i*)p_Data1Ptr); //Load the first 4 values of the left column
    data2Vector=_mm256_loadu_si256((__m256i*)p_Data2Ptr); //Load the first 4 values of the right column
    int changed_left=0;
    int full_hit=_mm256_movemask_pd((__m256d)(_mm256_cmpeq_epi64(data1Vector,data1Vector)));
   
    //Iterate as long as there are still values left in both columns
    while(p_Data1Ptr < endInPosL && p_Data2Ptr < endInPosR-4){

          /* Check all combinations for equality:
           * 1. Compare left and right, and store the result as a mask. Bitwise OR the masks of all combinations per iteration.
           * 2. Make a greater-than comparison between right and left. STore the result as a mask. Bitwise OR the masks of all combinations per iteration.
           * 3. Rotate the right relation by 64 bit
           */
        //for (int i=0;i<4;i++){
            mask = _mm256_movemask_pd((__m256d)(_mm256_cmpeq_epi64(data2Vector,data1Vector)));
            
            mask_greater_than =  _mm256_movemask_pd((__m256d)(_mm256_cmpgt_epi64(data1Vector,data2Vector)));
            //right=_mm256_permute4x64_epi64(right,57);
        //}
        
        //Store all matching values (use left relation)
        //compress_store256(outPos,mask,left);
        //Increase output position yb thenumber of results in this iteration
        //outPos=(__m256i*)(((uint64_t *)outPos)+__builtin_popcountl(mask));
             if (mask!=0) {
                *p_OutPtr=*p_Data1Ptr;//if keys are not unique, use a compressstore here
      
                p_OutPtr++;
             }
        if((mask_greater_than) == 0) { 
               p_Data1Ptr++;
               data1Vector = _mm256_set1_epi64x(*p_Data1Ptr);
               
            }else{
                if((mask_greater_than) == full_hit) { 
                    p_Data2Ptr += 4;
                    data2Vector = _mm256_loadu_si256(
                       (__m256i*)p_Data2Ptr
                    );
                    
                }else{
                    p_Data1Ptr++;
                    data1Vector = _mm256_set1_epi64x(*p_Data1Ptr);
                    p_Data2Ptr += __builtin_popcount(mask_greater_than);
                    data2Vector = _mm256_loadu_si256(
                       (__m256i*)p_Data2Ptr
                    );
                }
               
            }
        //Find out if we load the next vector from the right or the left column and load it -> check results of equality and greater-than comparison done earlier
        /*if ((mask | mask_greater_than) == 0) {
           inPosR++;
           right=_mm256_loadu_si256(inPosR);
           changed_left=0;
        }else{
            if ((mask ^ mask_greater_than) !=0){
                inPosR++;
                right=_mm256_loadu_si256(inPosR);
                changed_left=0; 
            }else{
           inPosL++;
           left=_mm256_loadu_si256(inPosL);
           changed_left=1;
            }
        }*/

        //Reset all masks for the next iteration

        
        
    }
    
    uint64_t * inPosL2 = (uint64_t *) p_Data1Ptr;
    uint64_t * inPosR2 = (uint64_t *) p_Data2Ptr;
    uint64_t * outPos2 = (uint64_t *) p_OutPtr;
    /*if (inPosL2-4 < (uint64_t *)endInPosL && changed_left==1){
         inPosL2-=4;
    }
    if ( (inPosR2-4 < (uint64_t *)endInPosR) && changed_left==0) {
        inPosR2-=4;
    }*/

    while(inPosL2 < (uint64_t *)endInPosL && inPosR2 < (uint64_t *)endInPosR) {
        if (changed_left) {trace( "[DEBUG] - intersect sequential tail" );}
        if(*inPosL2 < *inPosR2)
            inPosL2++;
        else if(*inPosR2 < *inPosL2)
            inPosR2++;
        else { // *inPosL == *inPosR
            *outPos2 = *inPosL2;
            outPos2++;
            inPosL2++;
            inPosR2++;
        }
    }
    

    p_OutPtr =  outPos2;
    
    const size_t outPosCount = ((uint64_t *)p_OutPtr - (uint64_t *)out_init);//How large is our result set?
    
    //Store output size in meta data of the output column
    outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
    return outPosCol; 
    
    
}
}

#endif /* INTERSECT_UNCOMPR_H */

