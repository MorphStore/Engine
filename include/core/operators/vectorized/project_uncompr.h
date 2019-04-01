/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   project_uncompr.h
 * Author: Annett
 *
 * Created on 29. März 2019, 13:06
 */

#ifndef PROJECT_UNCOMPR_H
#define PROJECT_UNCOMPR_H

#include <core/operators/interfaces/project.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>
#include <immintrin.h>

#include <cstdint>

namespace morphstore {
    
template<>
const column<uncompr_f> *
project<processing_style_t::vec128>(
        const column<uncompr_f> * const inDataCol,
        const column<uncompr_f> * const inPosCol
) {
    
    const size_t inPosCount = inPosCol->get_count_values();
     const uint64_t * const inData = inDataCol->get_data();
     const uint64_t * const inPos = inPosCol->get_data();
    
    const size_t inPosSize = inPosCol->get_size_used_byte();
    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(inPosSize);
    __m128i * outData = outDataCol->get_data();
    __m128i buffer;
    
    for(unsigned i = 0; i < inPosCount/2; i++) {
        //A gather could be faster here but requires loading a second register with the indexes in every loop
        //->Any opinions about that?
        buffer=_mm_set_epi64x(inData[inPos[i*2+1]],inData[inPos[i*2]]);
        _mm_store_si128(outData,buffer);
        outData++;
    }
    
    outDataCol->set_meta_data(inPosCount, inPosSize);
    
    return outDataCol;
}

template<>
const column<uncompr_f> *
project<processing_style_t::vec256>(
        const column<uncompr_f> * const inDataCol,
        const column<uncompr_f> * const inPosCol
) {
    
    const size_t inPosCount = inPosCol->get_count_values();
     const uint64_t * const inData = inDataCol->get_data();
     const uint64_t * const inPos = inPosCol->get_data();
    
    const size_t inPosSize = inPosCol->get_size_used_byte();
    // Exact allocation size (for uncompressed data).
    auto outDataCol = new column<uncompr_f>(inPosSize);
    __m256i * outData = outDataCol->get_data();
    __m256i buffer;
    
    
    for(size_t i = 0; i < inPosCount; i+=4) {
        //A gather could be faster here but requires loading a second register with the indexes in every loop
        //->Any opinions about that?
        buffer=_mm256_set_epi64x(inData[inPos[i+3]],inData[inPos[i+2]],inData[inPos[i+1]],inData[inPos[i]]);
        _mm256_store_si256(outData,buffer);
        outData++;
    }
    
    outDataCol->set_meta_data(inPosCount, inPosSize);
    
    return outDataCol;
}

}


#endif /* PROJECT_UNCOMPR_H */

