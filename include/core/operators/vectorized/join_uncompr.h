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
                processing_style_t::scalar,
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
    for(unsigned iL = 0; iL < inDataLCount/4; iL++){
        left=_mm256_load_si256(inDataL+iL);
        leftIDs=_mm256_set_epi64x(iL*4+3,iL*4+2,iL*4+1,iL*4);
        for(unsigned iR = 0; iR < inDataRCount/4; iR++){
           right=_mm256_load_si256(inDataR+iR);
           rightIDs=_mm256_set_epi64x(iR*4+3,iR*4+2,iR*4+1,iR*4);
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
    }
    }
    const size_t outSize = iOut * sizeof(uint64_t);
    outPosLCol->set_meta_data(iOut, outSize);
    outPosRCol->set_meta_data(iOut, outSize);
    
    return std::make_tuple(outPosLCol, outPosRCol);
    
}
}

#endif /* JOIN_UNCOMPR_H */

