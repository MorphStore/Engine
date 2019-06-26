/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   merg_uncompr.h
 * Author: Annett
 *
 * Created on 5. April 2019, 13:04
 */

#ifndef MERGE_UNCOMPR_H
#define MERGE_UNCOMPR_H


#include <core/operators/interfaces/merge.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/simd/avx2/extension_avx2.h>

#include <immintrin.h>
#include <cstdint>
#include <math.h> 

namespace morphstore {
    
template<>
const column<uncompr_f> *
merge_sorted<vector::avx2<vector::v256<uint64_t>>>(
        const column<uncompr_f> * const inPosLCol,
        const column<uncompr_f> * const inPosRCol
   
) {
    
    const uint64_t * inPosL;
    const uint64_t * inPosR;
  
    int LEnd;
    int REnd;
    
    //We hope that the larger column has the longest sequential runs, i.e. we get more sequential memory access
    if (inPosRCol->get_count_values() > inPosLCol->get_count_values()){
        inPosL = inPosLCol->get_data();
        inPosR = inPosRCol->get_data();
        LEnd = (int)inPosLCol->get_count_values();
        REnd = (int)inPosRCol->get_count_values();
    }else{
        inPosL = inPosRCol->get_data();
        inPosR = inPosLCol->get_data();
        LEnd = (int)inPosRCol->get_count_values();
        REnd = (int)inPosLCol->get_count_values();
    }

    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if the two input columns are disjoint.
    auto outPosCol = new column<uncompr_f>(
           
                    inPosLCol->get_size_used_byte() +
                    inPosRCol->get_size_used_byte()
          
    );
    
    uint64_t * outPos = outPosCol->get_data();
    const uint64_t * const initOutPos = outPos;
    __m256i left;
    __m256i right;

    
    int mask_gt=0;
    __m256i mask_gt_big;
    int mask_eq=0;
    int idx_right=0;
    int idx_left=0;
    right=_mm256_load_si256((__m256i*)inPosR);
    left=_mm256_set1_epi64x(inPosL[0]);

    while(idx_left< LEnd && idx_right < (REnd-4)) {
        
        mask_gt_big=_mm256_cmpgt_epi64(left,right);//left side greater than right side?     
        mask_eq=_mm256_movemask_pd((__m256d)_mm256_cmpeq_epi64(left,right));//left and right side equal?  
        mask_gt=_mm256_movemask_pd((__m256d)mask_gt_big);
        
        //if no values on left side are greater than on right side 
        if (mask_gt==0){
            if (mask_eq==0){ //avoid duplicates
                *outPos=_mm256_extract_epi64(left,0);//save current value from left side if it is not greater than the right side value
                outPos++;
            }
            idx_left++;
            left=_mm256_set1_epi64x((inPosL)[idx_left]);//broadcast next value from left side into left register
        }else {
            _mm256_maskstore_epi64((long long*)outPos, mask_gt_big, right);//save all values from right side, where left side is grater than right side -> a normal (u)store should work, too
        
            idx_right+= (__builtin_popcountl(mask_gt));//how many elements were that?
            outPos+=(__builtin_popcountl(mask_gt));
            right=_mm256_loadu_si256((__m256i*)((inPosR)+idx_right)); //(re)load next 4 elements (reloading all elements that were greater than on the left side)
        }
        
    }   

    //Do the remainder sequentially
    while(idx_left< LEnd && idx_right < REnd) {
      if((inPosL)[idx_left] < (inPosR)[idx_right]) {
            *outPos = (inPosL)[idx_left];
            idx_left++;
        }
        else if((inPosR)[idx_right] < (inPosL)[idx_left]) {
            *outPos = (inPosR)[idx_right];
            idx_right++;
        }
        else { // *inPosL == *inPosR
            *outPos = (inPosL)[idx_left];
            idx_left++;
            idx_right++;
        }
        outPos++;
    }
    
    
     // At this point, at least one of the operands has been fully consumed and
    // the other one might still contain data elements, which must be output.

   while (idx_left < LEnd){
        left=_mm256_loadu_si256((__m256i*)(inPosL+idx_left));
        _mm256_storeu_si256((__m256i*)outPos,left);

        idx_left=idx_left+4;
        outPos+=4;
     
    }
    if (idx_left>LEnd) outPos-=(idx_left-LEnd);//correct last output address if our last position is not divisible by 4
    
    while (idx_right < REnd){
        right=_mm256_loadu_si256((__m256i*)(inPosR+idx_right));
        _mm256_storeu_si256((__m256i*)outPos,right);

        idx_right=idx_right+4;
        outPos+=4;
       
    }
    
    if (idx_right>REnd) outPos-=(idx_right-REnd);//correct last output address if our last position is not divisible by 4
    
    
    
    //Copy rest, which didn't fit in a vetor register
        while (idx_left < LEnd){
            *outPos = idx_left;
             idx_left++;
             outPos++;
        }
         
        while (idx_right < REnd){
             *outPos = idx_right;
             idx_right++;
             outPos++;
        }
    
    const size_t outPosCount = ((uint64_t *)outPos - (uint64_t *)initOutPos);
    outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
    
    return outPosCol;
}

}

#endif /* MERGE_UNCOMPR_H */

