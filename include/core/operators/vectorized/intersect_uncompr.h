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
#include <core/utils/processing_style.h>
#include <immintrin.h>
#include <cstdint>

namespace morphstore {

template<>
const column<uncompr_f> *
intersect_sorted<processing_style_t::vec256>(
        const column<uncompr_f> * const inPosLCol,
        const column<uncompr_f> * const inPosRCol,
        const size_t outPosCountEstimate
) {
    const __m256i * inPosL = inPosLCol->get_data();
    const __m256i * inPosR = inPosRCol->get_data();
    const __m256i * const endInPosL = (__m256i*)((uint64_t*)inPosL + inPosLCol->get_count_values());
    const __m256i * const endInPosR = (__m256i*)((uint64_t*)inPosR + inPosRCol->get_count_values());
    
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if all positions in the smaller input
    // column are contained in the larger input column as well.
    auto outPosCol = new column<uncompr_f>(
            bool(outPosCountEstimate)
            // use given estimate
            ? (outPosCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : std::min(
                    inPosLCol->get_size_used_byte(),
                    inPosRCol->get_size_used_byte()
            )
    );
    __m256i * outPos = outPosCol->get_data();
    const __m256i * const initOutPos = outPos;

    __m256i left;
    __m256i right;
    int mask=0;
    int mask_greater_than=0;
    left=_mm256_loadu_si256(inPosL); //Load the first 4 values of the left column
    right=_mm256_loadu_si256(inPosR); //Load the first 4 values of the right column
    int changed_left=0;
   
    //Iterate as long as there are still values left in both columns
    while(inPosL < endInPosL && inPosR < endInPosR){

          /* Check all combinations for equality:
           * 1. Compare left and right, and store the result as a mask. Bitwise OR the masks of all combinations per iteration.
           * 2. Make a greater-than comparison between right and left. STore the result as a mask. Bitwise OR the masks of all combinations per iteration.
           * 3. Rotate the right relation by 64 bit
           */
        for (int i=0;i<4;i++){
            mask |= _mm256_movemask_pd((__m256d)(_mm256_cmpeq_epi64(right,left)));
            
            mask_greater_than |=  _mm256_movemask_pd((__m256d)(_mm256_cmpgt_epi64(right,left)));
            right=_mm256_permute4x64_epi64(right,57);
        }
        
        //Store all matching values (use left relation)
        compress_store256(outPos,mask,left);
        //Increase output position yb thenumber of results in this iteration
        outPos=(__m256i*)(((uint64_t *)outPos)+__builtin_popcountl(mask));
        
        //Find out if we load the next vector from the right or the left column and load it -> check results of equality and greater-than comparison done earlier
        if ((mask | mask_greater_than) == 0) {
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
        }

        //Reset all masks for the next iteration
        mask=0;
        mask_greater_than=0;
        
        
    }
    
    uint64_t * inPosL2 = (uint64_t *) inPosL;
    uint64_t * inPosR2 = (uint64_t *) inPosR;
    uint64_t * outPos2 = (uint64_t *) outPos;
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
    
    inPosL = (__m256i *) inPosL2;
    inPosR = (__m256i *) inPosR2;
    outPos = (__m256i *) outPos2;
    
    const size_t outPosCount = ((uint64_t *)outPos - (uint64_t *)initOutPos);//How large is our result set?
    
    //Store output size in meta data of the output column
    outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
    return outPosCol; 
    
    
}
}

#endif /* INTERSECT_UNCOMPR_H */

