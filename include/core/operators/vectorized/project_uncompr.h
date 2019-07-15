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
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/sse/extension_sse.h>

#include <immintrin.h>

#include <cstdint>

namespace morphstore {
    
template<>
struct project_t<
        vectorlib::sse<vectorlib::v128<uint64_t>>,
        uncompr_f,
        uncompr_f,
        uncompr_f
> {
    static
    const column<uncompr_f> *
    apply(
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

        unsigned i;
        for(i=0; i < inPosCount; i+=2) {
            //A gather could be faster here but requires loading a second register with the indexes in every loop
            //->Any opinions about that?
            buffer=_mm_set_epi64x(inData[inPos[i+1]],inData[inPos[i]]);
            _mm_store_si128(outData,buffer);
            outData++;
        }

        //Process the last elements (which do not fill up a whole register) sequentially
        unsigned k=i;
        uint64_t* oData=(uint64_t*) outData;
        for(i=k; i < inPosCount; i++) {
            oData[0] = inData[inPos[i]];
            oData++;
        }

        outDataCol->set_meta_data(inPosCount, inPosSize);

        return outDataCol;
    }
};

template<>
struct project_t<
        vectorlib::avx2<vectorlib::v256<uint64_t>>,
        uncompr_f,
        uncompr_f,
        uncompr_f
> {
    static
    const column<uncompr_f> *
    apply(
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

        unsigned i;
        for(i=0; i < inPosCount-4; i+=4) {
            //A gather could be faster here but requires loading a second register with the indexes in every loop
            //->Any opinions about that?
            buffer=_mm256_set_epi64x(inData[inPos[i+3]],inData[inPos[i+2]],inData[inPos[i+1]],inData[inPos[i]]);
            _mm256_store_si256(outData,buffer);
            outData++;
        }

        //Process the last elements (which do not fill up a whole register) sequentially
        unsigned k=i;
        uint64_t* oData=(uint64_t*) outData;
        for(i=k; i < inPosCount; i++) {
            *oData = inData[inPos[i]];
            oData++;
        }

        outDataCol->set_meta_data(inPosCount, inPosSize);

        return outDataCol;
    }
};

}

#endif /* PROJECT_UNCOMPR_H */