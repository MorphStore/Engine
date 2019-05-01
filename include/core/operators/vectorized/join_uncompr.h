/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   join_uncompr.h
 * Author: Annett
 *
 * Created on 3. April 2019, 12:29
 */

#ifndef JOIN_UNCOMPR_H
#define JOIN_UNCOMPR_H

#include <core/operators/interfaces/join.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>
#include <core/operators/vectorized/select_uncompr.h>

#include <cstdint>
#include <tuple>


namespace morphstore {
    
template<>
const std::tuple<
        const column<uncompr_f> *,
        const column<uncompr_f> *
>
nested_loop_join<processing_style_t::vec256>(
        const column<uncompr_f> * const inDataLCol,
        const column<uncompr_f> * const inDataRCol,
        const size_t outCountEstimate
) {
    const size_t inDataLCount = inDataLCol->get_count_values();
    const size_t inDataRCount = inDataRCol->get_count_values();
    
    // Ensure that the left column is the larger one, swap the input and output
    // column order if necessary.
    if(inDataLCount < inDataRCount) {
        auto outPosRL = nested_loop_join<
                processing_style_t::vec256,
                uncompr_f,
                uncompr_f
        >(
                inDataRCol,
                inDataLCol,
                outCountEstimate
        );
        return std::make_tuple(std::get<1>(outPosRL), std::get<0>(outPosRL));
    }
    
    const __m256i * const inDataL = inDataLCol->get_data();
    const __m256i * const inDataR = inDataRCol->get_data();
    
    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if the result is the cross product of
    // the two input columns.
    const size_t size = bool(outCountEstimate)
            // use given estimate
            ? (outCountEstimate * sizeof(uint64_t))
            // use pessimistic estimate
            : (inDataLCount * inDataRCount * sizeof(uint64_t));
    auto outPosLCol = new column<uncompr_f>(size);
    auto outPosRCol = new column<uncompr_f>(size);
    __m256i * outPosL = outPosLCol->get_data();
    __m256i * outPosR = outPosRCol->get_data();
    
    unsigned iOut = 0;
    __m256i cmpres;
    int mask=0;
    __m256i left;
    __m256i right;
    __m256i leftIDs;
    __m256i rightIDs;
    leftIDs=_mm256_set_epi64x(3,2,1,0);
    __m256i add=_mm256_set_epi64x(4,4,4,4);//!We will use this vector later to increase the IDs in every iteration
    
    //Iterate over all elements of the left column, 4 in each iteration because we can store 4 64-bit values in a 128 bit register
    for(unsigned iL = 0; iL < inDataLCount/4; iL++){
        //Load 4 values of the left relation into a vector register
        left=_mm256_load_si256(inDataL+iL);
     
        
        rightIDs=_mm256_set_epi64x(3,2,1,0);
        //Iterate over all elements of the right column, 4 in each iteration because we can store 4 64-bit values in a 128 bit register
        for(unsigned iR = 0; iR < inDataRCount/4; iR++){
            //Load 4 values of the right relation into a vector register
           right=_mm256_load_si256(inDataR+iR);
              
           /* Check all combinations for equality:
            * 1. Compare left and right
            * 2. Make a mask out of the result of step 1
            * 3. Store all IDs of the left relation, where the values matched with the right relation
            * 3. Store all IDs of the right relation, where the values matched with the left relation
            * 4. Increase the output address for the left relation by the number of results in this iteration
            * 5. Increase the output address for the right relation by the number of results in this iteration
            * 6. Rotate the right relation by 64 bit
            * 7. Rotate the indexes of the right relation by 64 bit
            * 8. Increas ethe output counter by thenumber of results in this iteration
            */
            for (int i=0; i<4;i++){
                cmpres=_mm256_cmpeq_epi64(left,right);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPosL,mask,leftIDs);
                compress_store256(outPosR,mask,rightIDs);
                outPosL=(__m256i*)(((uint64_t *)outPosL)+__builtin_popcountl(mask));
                outPosR=(__m256i*)(((uint64_t *)outPosR)+__builtin_popcountl(mask));
                right=_mm256_permute4x64_epi64(right,57); //rotate data to left
                rightIDs=_mm256_permute4x64_epi64(rightIDs,57); //rotate IDs to left
                iOut+=__builtin_popcountl(mask);
            }
           
           //Increase IDs for the right relation
           rightIDs=_mm256_add_epi64(rightIDs,add);
        }
        //Increase IDs for the left relation
        leftIDs=_mm256_add_epi64(leftIDs,add);
    }
    
    const size_t outSize = iOut * sizeof(uint64_t);//How large is our result set?
    
    //Store output size in meta data of the output columns
    outPosLCol->set_meta_data(iOut, outSize);
    outPosRCol->set_meta_data(iOut, outSize);
    
    return std::make_tuple(outPosLCol, outPosRCol);
    
}
}

#endif /* JOIN_UNCOMPR_H */

