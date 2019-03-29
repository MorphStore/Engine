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
    left=_mm256_loadu_si256(inPosL);
    right=_mm256_loadu_si256(inPosR);
    
    
    while(inPosL < endInPosL && inPosR < endInPosR){

        for (int i=0;i<4;i++){
            mask= (mask | _mm256_movemask_pd((__m256d)(_mm256_cmpeq_epi64(right,left))));
            right=_mm256_permute4x64_epi64(right,228);

            mask_greater_than = (mask_greater_than | _mm256_movemask_pd((__m256d)(_mm256_cmpgt_epi64(right,left))));
        }
        
        compress_store256(outPos,mask,left);
        outPos=(__m256i*)(((uint64_t *)outPos)+__builtin_popcountl(mask));
    
        if ((mask | mask_greater_than) == 0) {
           inPosR++;
           right=_mm256_loadu_si256(inPosR);
        }else{
           inPosL++;
           left=_mm256_loadu_si256(inPosL); 
        }

        mask=0;
        mask_greater_than=0;
        
        
    }
    
    const size_t outPosCount = ((uint64_t *)outPos - (uint64_t *)initOutPos);
    outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
    return outPosCol; 
    
    
}
}

#endif /* INTERSECT_UNCOMPR_H */

