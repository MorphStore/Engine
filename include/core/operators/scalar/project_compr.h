/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file project_compr.h
 * @brief Template specialization of the project-operator for input positions
 * columns compressed in the format static_vbp_f and all other input/outputs
 * uncompressed using the scalar processing style.
 * @todo Efficient implementations.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_COMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_COMPR_H

#include <core/operators/interfaces/project.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp_routines.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

#include <cstdint>
#include <immintrin.h>

namespace morphstore {
    
// @todo Reduce this code duplication.
// The following routines were copied from the unpacking routines of the
// vertical bit-packed format and adapted to facilitate the project operator.
    
// ************************************************************************
// Unpacking routines (with projection)
// ************************************************************************

// ------------------------------------------------------------------------
// Interfaces
// ------------------------------------------------------------------------

// Struct for partial template specialization.
template<
        processing_style_t t_ps,
        unsigned t_bw,
        unsigned t_step
>
struct unpack_with_project_t {
    static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
            const uint8_t * & inData8,
            const uint8_t * & inPos8,
            uint8_t * & outData8,
            size_t countOutData64
    ) = delete;
};

// Convenience function.
template<
        processing_style_t t_ps,
        unsigned t_bw,
        unsigned t_step
>
MSV_CXX_ATTRIBUTE_FORCE_INLINE void unpack_with_project(
        const uint8_t * & inData8,
        const uint8_t * & inPos8,
        uint8_t * & outData8,
        size_t countOutData64
) {
    unpack_with_project_t<t_ps, t_bw, t_step>::apply(inData8, inPos8, outData8, countOutData64);
}


// ------------------------------------------------------------------------
// Template specializations.
// ------------------------------------------------------------------------
    
// Generic w.r.t. the step width. Hopefully the compiler unrolls the loops.
template<unsigned t_bw, unsigned t_step>
struct unpack_with_project_t<
        processing_style_t::scalar,
        t_bw,
        t_step
> {
    static MSV_CXX_ATTRIBUTE_FORCE_INLINE void apply(
            const uint8_t * & inData8,
            const uint8_t * & inPos8,
            uint8_t * & outData8,
            size_t countOutData64
    ) {
        const uint64_t * inData64 = reinterpret_cast<const uint64_t *>(inData8);
        const uint64_t * inPos64 = reinterpret_cast<const uint64_t *>(inPos8);
        uint64_t * outData64 = reinterpret_cast<uint64_t *>(outData8);

        const size_t countBits = std::numeric_limits<uint64_t>::digits;
        const uint64_t mask = bitwidth_max<uint64_t>(t_bw);;

        // This variant uses a store instruction at only one point.
        uint64_t nextOut[t_step];
        for(unsigned i = 0; i < t_step; i++)
            nextOut[i] = 0;
        unsigned bitpos = countBits + t_bw;
        const uint64_t * const endOutData64 = outData64 + countOutData64;
        while(outData64 < endOutData64) {
            uint64_t tmp[t_step];
            if(bitpos == countBits + t_bw) {
                for(unsigned i = 0; i < t_step; i++) {
                    tmp[i] = *inPos64++;
                    nextOut[i] = mask & tmp[i];
                }
                bitpos = t_bw;
            }
            else { // bitpos > countBits && bitpos < countBits + t_bw
                for(unsigned i = 0; i < t_step; i++) {
                    tmp[i] = *inPos64++;
                    nextOut[i] = mask & ((tmp[i] << (countBits - bitpos + t_bw)) | nextOut[i]);
                }
                bitpos = bitpos - countBits;
            }
            while(bitpos <= countBits) {
                for(unsigned i = 0; i < t_step; i++) {
                    *outData64++ = inData64[nextOut[i]];
                    nextOut[i] = mask & (tmp[i] >> bitpos);
                }
                bitpos += t_bw;
            }
        }

        inData8 = reinterpret_cast<const uint8_t *>(inData64);
        inPos8 = reinterpret_cast<const uint8_t *>(inPos64);
        outData8 = reinterpret_cast<uint8_t *>(outData64);
    }
};
    
// ************************************************************************
// Project operator on static_vbp_f-compressed positions
// ************************************************************************
    
// Generic implementation; the magic happens in unpack_with_project.
template<processing_style_t t_ps, unsigned t_bw, unsigned t_step>
struct project_t<
        t_ps,
        uncompr_f,
        uncompr_f,
        static_vbp_f<t_bw, t_step>
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<static_vbp_f<t_bw, t_step> > * const inPosCol
    ) {
        const size_t inPosCount64 = inPosCol->get_count_values();
        const uint8_t * inData8 = inDataCol->get_data();
        const uint8_t * inPos8 = inPosCol->get_data();

        // Exact allocation size (for uncompressed data).
        const size_t outSize = uncompr_f::get_size_max_byte(inPosCount64);
        auto outDataCol = new column<uncompr_f>(outSize);
        uint8_t * outData8 = outDataCol->get_data();

        unpack_with_project<t_ps, t_bw, t_step>(
                inData8, inPos8, outData8, inPosCount64
        );

        outDataCol->set_meta_data(inPosCount64, outSize);

        return outDataCol;
    }
};

// Some special cases manually tailored to specific bit widths and steps.
#if 1
template<>
struct project_t<
        processing_style_t::scalar,
        uncompr_f,
        uncompr_f,
        static_vbp_f<32, sizeof(__m128i) / sizeof(uint64_t)>
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<static_vbp_f<32, sizeof(__m128i) / sizeof(uint64_t)> > * const inPosCol
    ) {
        const size_t inPosCount = inPosCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        const uint32_t * inPos = inPosCol->get_data();
        const uint32_t * const endInPos = inPos + inPosCount;

        // Exact allocation size (for uncompressed data).
        const size_t outDataSize = uncompr_f::get_size_max_byte(inPosCount);
        auto outDataCol = new column<uncompr_f>(outDataSize);
        uint64_t * outData = outDataCol->get_data();

        while(inPos < endInPos) {
#if 1
            // Baseline.
            for(unsigned i = 0; i < 2; i++) {
                *outData++ = inData[inPos[i    ]];
                *outData++ = inData[inPos[i + 2]];
            }
            inPos += 4;
#elif 1
            // Equally fast.
            for(unsigned i = 0; i < 8; i++) {
                outData[2 * i    ] = inData[inPos[i    ]];
                outData[2 * i + 1] = inData[inPos[i + 8]];
            }
            inPos += 16;
            outData += 16;
#else
            // Equally fast.
            outData[ 0] = inData[inPos[ 0]];
            outData[ 1] = inData[inPos[ 8]];
            outData[ 2] = inData[inPos[ 1]];
            outData[ 3] = inData[inPos[ 9]];
            outData[ 4] = inData[inPos[ 2]];
            outData[ 5] = inData[inPos[10]];
            outData[ 6] = inData[inPos[ 3]];
            outData[ 7] = inData[inPos[11]];
            outData[ 8] = inData[inPos[ 4]];
            outData[ 9] = inData[inPos[12]];
            outData[10] = inData[inPos[ 5]];
            outData[11] = inData[inPos[13]];
            outData[12] = inData[inPos[ 6]];
            outData[13] = inData[inPos[14]];
            outData[14] = inData[inPos[ 7]];
            outData[15] = inData[inPos[15]];
            inPos += 16;
            outData += 16;
#endif
        }

        outDataCol->set_meta_data(inPosCount, outDataSize);

        return outDataCol;
    }
};
    
template<>
struct project_t<
        processing_style_t::scalar,
        uncompr_f,
        uncompr_f,
        static_vbp_f<16, sizeof(__m128i) / sizeof(uint64_t)>
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<static_vbp_f<16,sizeof(__m128i) / sizeof(uint64_t)> > * const inPosCol
    ) {
        const size_t inPosCount = inPosCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        const uint16_t * inPos = inPosCol->get_data();
        const uint16_t * const endInPos = inPos + inPosCount;

        // Exact allocation size (for uncompressed data).
        const size_t outDataSize = uncompr_f::get_size_max_byte(inPosCount);
        auto outDataCol = new column<uncompr_f>(outDataSize);
        uint64_t * outData = outDataCol->get_data();

        while(inPos < endInPos) {
#if 1
            // Baseline.
            for(unsigned i = 0; i < 4; i++) {
                *outData++ = inData[inPos[i    ]];
                *outData++ = inData[inPos[i + 4]];
            }
            inPos += 8;
#elif 1
            // Equally fast.
            for(unsigned i = 0; i < 8; i++) {
                outData[2 * i    ] = inData[inPos[i    ]];
                outData[2 * i + 1] = inData[inPos[i + 8]];
            }
            inPos += 16;
            outData += 16;
#else
            // Equally fast.
            outData[ 0] = inData[inPos[ 0]];
            outData[ 1] = inData[inPos[ 8]];
            outData[ 2] = inData[inPos[ 1]];
            outData[ 3] = inData[inPos[ 9]];
            outData[ 4] = inData[inPos[ 2]];
            outData[ 5] = inData[inPos[10]];
            outData[ 6] = inData[inPos[ 3]];
            outData[ 7] = inData[inPos[11]];
            outData[ 8] = inData[inPos[ 4]];
            outData[ 9] = inData[inPos[12]];
            outData[10] = inData[inPos[ 5]];
            outData[11] = inData[inPos[13]];
            outData[12] = inData[inPos[ 6]];
            outData[13] = inData[inPos[14]];
            outData[14] = inData[inPos[ 7]];
            outData[15] = inData[inPos[15]];
            inPos += 16;
            outData += 16;
#endif
        }

        outDataCol->set_meta_data(inPosCount, outDataSize);

        return outDataCol;
    }
};
    
template<>
struct project_t<
        processing_style_t::scalar,
        uncompr_f,
        uncompr_f,
        static_vbp_f<8, sizeof(__m128i) / sizeof(uint64_t)>
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<static_vbp_f<8, sizeof(__m128i) / sizeof(uint64_t)> > * const inPosCol
    ) {
        const size_t inPosCount = inPosCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        const uint8_t * inPos = inPosCol->get_data();
        const uint8_t * const endInPos = inPos + inPosCount;

        // Exact allocation size (for uncompressed data).
        const size_t outDataSize = uncompr_f::get_size_max_byte(inPosCount);
        auto outDataCol = new column<uncompr_f>(outDataSize);
        uint64_t * outData = outDataCol->get_data();

        while(inPos < endInPos) {
#if 1
            // Baseline.
            for(unsigned i = 0; i < 8; i++) {
                *outData++ = inData[inPos[i    ]];
                *outData++ = inData[inPos[i + 8]];
            }
            inPos += 16;
#elif 1
            // Equally fast.
            for(unsigned i = 0; i < 8; i++) {
                outData[2 * i    ] = inData[inPos[i    ]];
                outData[2 * i + 1] = inData[inPos[i + 8]];
            }
            inPos += 16;
            outData += 16;
#else
            // Equally fast.
            outData[ 0] = inData[inPos[ 0]];
            outData[ 1] = inData[inPos[ 8]];
            outData[ 2] = inData[inPos[ 1]];
            outData[ 3] = inData[inPos[ 9]];
            outData[ 4] = inData[inPos[ 2]];
            outData[ 5] = inData[inPos[10]];
            outData[ 6] = inData[inPos[ 3]];
            outData[ 7] = inData[inPos[11]];
            outData[ 8] = inData[inPos[ 4]];
            outData[ 9] = inData[inPos[12]];
            outData[10] = inData[inPos[ 5]];
            outData[11] = inData[inPos[13]];
            outData[12] = inData[inPos[ 6]];
            outData[13] = inData[inPos[14]];
            outData[14] = inData[inPos[ 7]];
            outData[15] = inData[inPos[15]];
            inPos += 16;
            outData += 16;
#endif
        }

        outDataCol->set_meta_data(inPosCount, outDataSize);

        return outDataCol;
    }
};
    
template<>
struct project_t<
        processing_style_t::scalar,
        uncompr_f,
        uncompr_f,
        static_vbp_f<4, sizeof(__m128i) / sizeof(uint64_t)>
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<static_vbp_f<4, sizeof(__m128i) / sizeof(uint64_t)> > * const inPosCol
    ) {
        const size_t inPosCount = inPosCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        const uint8_t * inPos = inPosCol->get_data();
        const uint8_t * const endInPos = inPos + inPosCount / 2;

        // Exact allocation size (for uncompressed data).
        const size_t outDataSize = uncompr_f::get_size_max_byte(inPosCount);
        auto outDataCol = new column<uncompr_f>(outDataSize);
        uint64_t * outData = outDataCol->get_data();

        while(inPos < endInPos) {
            // Baseline.
            for(unsigned i = 0; i < 8; i++) {
                *outData++ = inData[ inPos[i    ] & 0x0f];
                *outData++ = inData[ inPos[i + 8] & 0x0f];
                *outData++ = inData[(inPos[i    ] & 0xf0) >> 4];
                *outData++ = inData[(inPos[i + 8] & 0xf0) >> 4];
            }
            inPos += 16;
        }

        outDataCol->set_meta_data(inPosCount, outDataSize);

        return outDataCol;
    }
};
    
template<>
struct project_t<
        processing_style_t::scalar,
        uncompr_f,
        uncompr_f,
        static_vbp_f<2, sizeof(__m128i) / sizeof(uint64_t)>
> {
    static
    const column<uncompr_f> *
    apply(
            const column<uncompr_f> * const inDataCol,
            const column<static_vbp_f<2, sizeof(__m128i) / sizeof(uint64_t)> > * const inPosCol
    ) {
        const size_t inPosCount = inPosCol->get_count_values();
        const uint64_t * const inData = inDataCol->get_data();
        const uint8_t * inPos = inPosCol->get_data();
        const uint8_t * const endInPos = inPos + inPosCount / 4;

        // Exact allocation size (for uncompressed data).
        const size_t outDataSize = uncompr_f::get_size_max_byte(inPosCount);
        auto outDataCol = new column<uncompr_f>(outDataSize);
        uint64_t * outData = outDataCol->get_data();

        while(inPos < endInPos) {
            // Baseline.
            for(unsigned i = 0; i < 8; i++) {
                *outData++ = inData[ inPos[i    ] & 0b00000011];
                *outData++ = inData[ inPos[i + 8] & 0b00000011];
                *outData++ = inData[(inPos[i    ] & 0b00001100) >> 2];
                *outData++ = inData[(inPos[i + 8] & 0b00001100) >> 2];
                *outData++ = inData[(inPos[i    ] & 0b00110000) >> 4];
                *outData++ = inData[(inPos[i + 8] & 0b00110000) >> 4];
                *outData++ = inData[(inPos[i    ] & 0b11000000) >> 6];
                *outData++ = inData[(inPos[i + 8] & 0b11000000) >> 6];
            }
            inPos += 16;
        }

        outDataCol->set_meta_data(inPosCount, outDataSize);

        return outDataCol;
    }
};
#endif

}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_PROJECT_COMPR_H