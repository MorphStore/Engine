

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
 * @file bitmap.h
 * @brief Bitmap (BM) intermediate representation with morphing-operators and other facilities.
 * @todo Implement decompress_and_batch_process, etc.
 */

#ifndef MORPHSTORE_CORE_MORPHING_INTERMEDIATES_BITMAP_H
#define MORPHSTORE_CORE_MORPHING_INTERMEDIATES_BITMAP_H

#include <core/morphing/format.h>
#include <core/morphing/intermediates/representation.h>
#include <core/morphing/morph_batch.h>
#include <core/morphing/morph.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {

    using namespace vectorlib;

    /**
    * @brief Bitmap representation/format with some inner-format.
    *
    *        An IR-type consists of an inner-format (e.g. uncompr_f, rle_f, etc.) and its logical representation (BM, PL).
    *        Generally: IR<format> , e.g. bitmap<rle_f> (run-length-encoded bitmap)
    *
    *        By default, the inner-format is uncompr_f so that we can just write e.g. 'position_list' instead
    *        of 'position_list<uncompr_f>' (>= c++17). Otherwise (i.e. < c++17), we have to write 'position_list<>'.
    *
    */
    template<class inner_format_t = uncompr_f>
    struct bitmap_f : public intermediate_representation, public format {
        using t_inner_f = inner_format_t;

        static_assert(
                std::is_base_of<representation, t_inner_f>::value,
                "bitmap_f: the template parameter t_inner_f must be a subclass of representation"
        );

        static size_t get_size_max_byte(size_t p_CountValues) {
            return t_inner_f::get_size_max_byte(p_CountValues);
        }

        static const size_t m_BlockSize = t_inner_f::m_BlockSize;

        // workaround for outer-type deduction, more precisely to get the underlying IR, used in representation.h
        static const intermediate_type ir_type = {intermediate_type::bitmap};

        // Generally used in write_iterator to allocate transformation buffer (see write_iterator_IR.h):
        // if every bit is set (pessimistic), 1 bitmap word can generate 64 positions => 32 * 64 * sizeof(uint64_t) = 16ki bytes (Lx-cache-resident)
        // -> we currently process 2048 elements in batches
        static const size_t trans_buf_cnt = 32;
    };

    // ------------------------------------------------------------------------
    // BM - Compression
    // ------------------------------------------------------------------------

    // ------------------------------ batch-level ------------------------------
    template<class t_vector_extension, class inner_format>
    struct morph_batch_t<t_vector_extension, bitmap_f<inner_format>, bitmap_f<uncompr_f> >{
        static void apply(
                const uint8_t * & p_In8,
                uint8_t * & p_Out8,
                size_t p_CountLog
        ) {
            using dest_f = inner_format;
            morph_batch<t_vector_extension, dest_f, uncompr_f>(p_In8, p_Out8, p_CountLog);
        }
    };

    // ------------------------------ column-level ------------------------------
    template<class t_vector_extension, class inner_format>
    struct morph_t<t_vector_extension, bitmap_f<inner_format>, bitmap_f<uncompr_f> >{
        using dest_format = inner_format;

        static
        const column< bitmap_f<dest_format> > *
        apply(
                const column< bitmap_f<uncompr_f> > * inCol
        ) {
            // return IR-column: cast from inner_format -> bitmap_f<inner_format>
            return
                    reinterpret_cast< const column< bitmap_f<dest_format> > * >
                    (
                            morph<t_vector_extension, dest_format, uncompr_f>(
                                    // cast inCol to uncompr_f (from bitmap_f<uncompr_f>)
                                    reinterpret_cast<const column<uncompr_f> *>(inCol)
                            )
                    );
        }
    };

    // ------------------------------------------------------------------------
    // BM - Decompression
    // ------------------------------------------------------------------------

    // ------------------------------ batch-level ------------------------------
    template<class t_vector_extension, class inner_format>
    struct morph_batch_t<t_vector_extension, bitmap_f<uncompr_f>, bitmap_f<inner_format> >{
        static void apply(
                const uint8_t * & p_In8,
                uint8_t * & p_Out8,
                size_t p_CountLog
        ) {
            using src_f = inner_format;
            morph_batch<t_vector_extension, uncompr_f, src_f>(p_In8, p_Out8, p_CountLog);
        }
    };

    // ------------------------------ column-level ------------------------------

    template<class t_vector_extension, class inner_format>
    struct morph_t<t_vector_extension, bitmap_f<uncompr_f>, bitmap_f<inner_format> >{
        using src_format = inner_format;

        static
        const column< bitmap_f<uncompr_f> > *
        apply(
                const column< bitmap_f<src_format> > * inCol
        ) {
            // return IR-column: cast from inner_format -> bitmap_f<inner_format>
            return
                    reinterpret_cast< const column< bitmap_f<uncompr_f> > * >
                    (
                            morph<t_vector_extension, uncompr_f, src_format>(
                                    // cast inCol to uncompr_f (from bitmap_f<uncompr_f>)
                                    reinterpret_cast<const column<src_format> *>(inCol)
                            )
                    );
        }
    };

    // ------------------------------------------------------------------------
    // Morph: bitmap_f<uncompr_f> --> uncompr_f & vice versa
    // ------------------------------------------------------------------------

    // Need these additional template specialization for that case, otherwise 'error: use of deleted function ...'
    // in morph_batch-interface: just a wrapper to morph_batch<ve, uncompr_f, uncompr_f>
    template<class t_vector_extension>
    struct morph_batch_t<t_vector_extension, uncompr_f, bitmap_f<uncompr_f> >{
        static void apply(
                const uint8_t * & p_In8,
                uint8_t * & p_Out8,
                size_t p_CountLog
        ) {
            morph_batch<t_vector_extension, uncompr_f, uncompr_f>(p_In8, p_Out8, p_CountLog);
        }
    };

    template<class t_vector_extension>
    struct morph_batch_t<t_vector_extension, bitmap_f<uncompr_f>, uncompr_f >{
        static void apply(
                const uint8_t * & p_In8,
                uint8_t * & p_Out8,
                size_t p_CountLog
        ) {
            morph_batch<t_vector_extension, uncompr_f, uncompr_f>(p_In8, p_Out8, p_CountLog);
        }
    };

    /**
     * @brief Bitmap processing state to store current characteristics (bit-position and active bitmap-word)
     *        that are especially needed in a bm-operator's processing unit (operator-scope),
     *        e.g. in vectorized processing when changing from vectorized to remainder part.
     *
     *        Assumption: bitmap is uncompressed
     *
     */
    struct bitmap_processing_state_t {

        uint64_t m_active_word; // current encoded word within the bitmap (64-bit integer)
        uint64_t m_bitPos; // current bit-position in m_active_word -> to know where to proceed

        bitmap_processing_state_t(uint64_t p_word, uint64_t p_bitPos) : m_active_word(p_word), m_bitPos(p_bitPos)
        {
            //
        };
    };

    // ************************************************************************
    // Bitmap general characteristics for processing
    // ************************************************************************

    /**
     * @brief Bitmap lookup tables to decode a sequence of bits into positions (needed for transformation to PL)
     *
     *        The tables are specialized according to the element-count of the vector-extension
     *              (   SSE     [ using uint64_t: element_count = 2 ],
     *                  AVX/2   [ using uint64_t: element_count = 4 ],
     *                  AVX512  [ using uint64_t: element_count = 8 ] ).
     *
     *   Note: The lookup-tables start encoding from 1 (not 0) because as we start counting from index 0 (bit-position),
     *         encodings for 1 and 0 would have the same encoding, e.g. in element_count = 4 -> {0, 0, 0, 0}
     *         => ambiguity (we don't need a 0-encoding as we skip words = 0)
     */

    // primary template
    template <size_t element_count, typename T>
    struct bitmap_lookup_tables{
        static_assert(
                std::is_same<uint64_t, T>::value,
        "bitmap processing is currently only supported for uint64_t."
        );
        static_assert(
        (element_count == 2 ) || (element_count == 4) || (element_count == 8),
        "bitmap lookup-tables are only supported for 2-, 4-, 8-elements-at-a-time"
        );
    };

    // full template specialization for vector_element_count = 2 and base_t = uint64_t, e.g. in sse4 v128
    template <>
    struct bitmap_lookup_tables<2, uint64_t>{
        static const uint64_t mask = 0x0000000000000003;
        static constexpr uint64_t lookup_table[3][2] = {
            { 0, 0 }, // (01)
            { 1, 0 }, // (10)
            { 0, 1 }, // (11)
        };
        static const uint64_t * get_positions(uint64_t in){
            return lookup_table[in];
        }
    };
    // <= -std=c++14: we have to provide a definition for any static constexpr data members that are odr-use;
    //                otherwise undefined reference error
    constexpr uint64_t bitmap_lookup_tables<2, uint64_t>::lookup_table[][2];

    // full template specialization for vector_element_count = 4 and base_t = uint64_t, e.g. in avx256
    template <>
    struct bitmap_lookup_tables<4, uint64_t>{
        static const uint64_t mask = 0x000000000000000F;
        static constexpr uint64_t lookup_table[15][4] = {
            { 0, 0, 0, 0 }, // 0x1 (0001)
            { 1, 0, 0, 0 }, // 0x2 (0010)
            { 0, 1, 0, 0 }, // 0x3 (0011)
            { 2, 0, 0, 0 }, // 0x4 (0100)
            { 0, 2, 0, 0 }, // 0x5 (0101)
            { 1, 2, 0, 0 }, // 0x6 (0110)
            { 0, 1, 2, 0 }, // 0x7 (0111)
            { 3, 0, 0, 0 }, // 0x8 (1000)
            { 0, 3, 0, 0 }, // 0x9 (1001)
            { 1, 3, 0, 0 }, // 0xA (1010)
            { 0, 1, 3, 0 }, // 0xB (1011)
            { 2, 3, 0, 0 }, // 0xC (1100)
            { 0, 2, 3, 0 }, // 0xD (1101)
            { 1, 2, 3, 0 }, // 0xE (1110)
            { 0, 1, 2, 3 }, // 0xF (1111)
        };
        static const uint64_t * get_positions(uint64_t in){
            return lookup_table[in];
        }
    };
    // <= -std=c++14: we have to provide a definition for any static constexpr data members that are odr-use;
    //                otherwise undefined reference error
    constexpr uint64_t bitmap_lookup_tables<4, uint64_t>::lookup_table[][4];

    // full template specialization for vector_element_count = 8 and base_t = uint64_t, e.g. in avx512
    template <>
    struct bitmap_lookup_tables<8, uint64_t>{
        static const uint64_t mask = 0x00000000000000FF;
        static constexpr uint64_t lookup_table[255][8] = {
            { 0, 0, 0, 0, 0, 0, 0, 0 }, /* 0x01 (00000001) */
            { 1, 0, 0, 0, 0, 0, 0, 0 }, /* 0x02 (00000010) */
            { 0, 1, 0, 0, 0, 0, 0, 0 }, /* 0x03 (00000011) */
            { 2, 0, 0, 0, 0, 0, 0, 0 }, /* 0x04 (00000100) */
            { 0, 2, 0, 0, 0, 0, 0, 0 }, /* 0x05 (00000101) */
            { 1, 2, 0, 0, 0, 0, 0, 0 }, /* 0x06 (00000110) */
            { 0, 1, 2, 0, 0, 0, 0, 0 }, /* 0x07 (00000111) */
            { 3, 0, 0, 0, 0, 0, 0, 0 }, /* 0x08 (00001000) */
            { 0, 3, 0, 0, 0, 0, 0, 0 }, /* 0x09 (00001001) */
            { 1, 3, 0, 0, 0, 0, 0, 0 }, /* 0x0A (00001010) */
            { 0, 1, 3, 0, 0, 0, 0, 0 }, /* 0x0B (00001011) */
            { 2, 3, 0, 0, 0, 0, 0, 0 }, /* 0x0C (00001100) */
            { 0, 2, 3, 0, 0, 0, 0, 0 }, /* 0x0D (00001101) */
            { 1, 2, 3, 0, 0, 0, 0, 0 }, /* 0x0E (00001110) */
            { 0, 1, 2, 3, 0, 0, 0, 0 }, /* 0x0F (00001111) */
            { 4, 0, 0, 0, 0, 0, 0, 0 }, /* 0x10 (00010000) */
            { 0, 4, 0, 0, 0, 0, 0, 0 }, /* 0x11 (00010001) */
            { 1, 4, 0, 0, 0, 0, 0, 0 }, /* 0x12 (00010010) */
            { 0, 1, 4, 0, 0, 0, 0, 0 }, /* 0x13 (00010011) */
            { 2, 4, 0, 0, 0, 0, 0, 0 }, /* 0x14 (00010100) */
            { 0, 2, 4, 0, 0, 0, 0, 0 }, /* 0x15 (00010101) */
            { 1, 2, 4, 0, 0, 0, 0, 0 }, /* 0x16 (00010110) */
            { 0, 1, 2, 4, 0, 0, 0, 0 }, /* 0x17 (00010111) */
            { 3, 4, 0, 0, 0, 0, 0, 0 }, /* 0x18 (00011000) */
            { 0, 3, 4, 0, 0, 0, 0, 0 }, /* 0x19 (00011001) */
            { 1, 3, 4, 0, 0, 0, 0, 0 }, /* 0x1A (00011010) */
            { 0, 1, 3, 4, 0, 0, 0, 0 }, /* 0x1B (00011011) */
            { 2, 3, 4, 0, 0, 0, 0, 0 }, /* 0x1C (00011100) */
            { 0, 2, 3, 4, 0, 0, 0, 0 }, /* 0x1D (00011101) */
            { 1, 2, 3, 4, 0, 0, 0, 0 }, /* 0x1E (00011110) */
            { 0, 1, 2, 3, 4, 0, 0, 0 }, /* 0x1F (00011111) */
            { 5, 0, 0, 0, 0, 0, 0, 0 }, /* 0x20 (00100000) */
            { 0, 5, 0, 0, 0, 0, 0, 0 }, /* 0x21 (00100001) */
            { 1, 5, 0, 0, 0, 0, 0, 0 }, /* 0x22 (00100010) */
            { 0, 1, 5, 0, 0, 0, 0, 0 }, /* 0x23 (00100011) */
            { 2, 5, 0, 0, 0, 0, 0, 0 }, /* 0x24 (00100100) */
            { 0, 2, 5, 0, 0, 0, 0, 0 }, /* 0x25 (00100101) */
            { 1, 2, 5, 0, 0, 0, 0, 0 }, /* 0x26 (00100110) */
            { 0, 1, 2, 5, 0, 0, 0, 0 }, /* 0x27 (00100111) */
            { 3, 5, 0, 0, 0, 0, 0, 0 }, /* 0x28 (00101000) */
            { 0, 3, 5, 0, 0, 0, 0, 0 }, /* 0x29 (00101001) */
            { 1, 3, 5, 0, 0, 0, 0, 0 }, /* 0x2A (00101010) */
            { 0, 1, 3, 5, 0, 0, 0, 0 }, /* 0x2B (00101011) */
            { 2, 3, 5, 0, 0, 0, 0, 0 }, /* 0x2C (00101100) */
            { 0, 2, 3, 5, 0, 0, 0, 0 }, /* 0x2D (00101101) */
            { 1, 2, 3, 5, 0, 0, 0, 0 }, /* 0x2E (00101110) */
            { 0, 1, 2, 3, 5, 0, 0, 0 }, /* 0x2F (00101111) */
            { 4, 5, 0, 0, 0, 0, 0, 0 }, /* 0x30 (00110000) */
            { 0, 4, 5, 0, 0, 0, 0, 0 }, /* 0x31 (00110001) */
            { 1, 4, 5, 0, 0, 0, 0, 0 }, /* 0x32 (00110010) */
            { 0, 1, 4, 5, 0, 0, 0, 0 }, /* 0x33 (00110011) */
            { 2, 4, 5, 0, 0, 0, 0, 0 }, /* 0x34 (00110100) */
            { 0, 2, 4, 5, 0, 0, 0, 0 }, /* 0x35 (00110101) */
            { 1, 2, 4, 5, 0, 0, 0, 0 }, /* 0x36 (00110110) */
            { 0, 1, 2, 4, 5, 0, 0, 0 }, /* 0x37 (00110111) */
            { 3, 4, 5, 0, 0, 0, 0, 0 }, /* 0x38 (00111000) */
            { 0, 3, 4, 5, 0, 0, 0, 0 }, /* 0x39 (00111001) */
            { 1, 3, 4, 5, 0, 0, 0, 0 }, /* 0x3A (00111010) */
            { 0, 1, 3, 4, 5, 0, 0, 0 }, /* 0x3B (00111011) */
            { 2, 3, 4, 5, 0, 0, 0, 0 }, /* 0x3C (00111100) */
            { 0, 2, 3, 4, 5, 0, 0, 0 }, /* 0x3D (00111101) */
            { 1, 2, 3, 4, 5, 0, 0, 0 }, /* 0x3E (00111110) */
            { 0, 1, 2, 3, 4, 5, 0, 0 }, /* 0x3F (00111111) */
            { 6, 0, 0, 0, 0, 0, 0, 0 }, /* 0x40 (01000000) */
            { 0, 6, 0, 0, 0, 0, 0, 0 }, /* 0x41 (01000001) */
            { 1, 6, 0, 0, 0, 0, 0, 0 }, /* 0x42 (01000010) */
            { 0, 1, 6, 0, 0, 0, 0, 0 }, /* 0x43 (01000011) */
            { 2, 6, 0, 0, 0, 0, 0, 0 }, /* 0x44 (01000100) */
            { 0, 2, 6, 0, 0, 0, 0, 0 }, /* 0x45 (01000101) */
            { 1, 2, 6, 0, 0, 0, 0, 0 }, /* 0x46 (01000110) */
            { 0, 1, 2, 6, 0, 0, 0, 0 }, /* 0x47 (01000111) */
            { 3, 6, 0, 0, 0, 0, 0, 0 }, /* 0x48 (01001000) */
            { 0, 3, 6, 0, 0, 0, 0, 0 }, /* 0x49 (01001001) */
            { 2, 3, 6, 0, 0, 0, 0, 0 }, /* 0x4A (01001010) */
            { 0, 1, 3, 6, 0, 0, 0, 0 }, /* 0x4B (01001011) */
            { 2, 3, 6, 0, 0, 0, 0, 0 }, /* 0x4C (01001100) */
            { 0, 2, 3, 6, 0, 0, 0, 0 }, /* 0x4D (01001101) */
            { 1, 2, 3, 6, 0, 0, 0, 0 }, /* 0x4E (01001110) */
            { 0, 1, 2, 3, 6, 0, 0, 0 }, /* 0x4F (01001111) */
            { 4, 6, 0, 0, 0, 0, 0, 0 }, /* 0x50 (01010000) */
            { 0, 4, 6, 0, 0, 0, 0, 0 }, /* 0x51 (01010001) */
            { 1, 4, 6, 0, 0, 0, 0, 0 }, /* 0x52 (01010010) */
            { 0, 1, 4, 6, 0, 0, 0, 0 }, /* 0x53 (01010011) */
            { 2, 4, 6, 0, 0, 0, 0, 0 }, /* 0x54 (01010100) */
            { 0, 2, 4, 6, 0, 0, 0, 0 }, /* 0x55 (01010101) */
            { 1, 2, 4, 6, 0, 0, 0, 0 }, /* 0x56 (01010110) */
            { 0, 1, 2, 4, 6, 0, 0, 0 }, /* 0x57 (01010111) */
            { 3, 4, 6, 0, 0, 0, 0, 0 }, /* 0x58 (01011000) */
            { 0, 3, 4, 6, 0, 0, 0, 0 }, /* 0x59 (01011001) */
            { 1, 3, 4, 6, 0, 0, 0, 0 }, /* 0x5A (01011010) */
            { 0, 1, 3, 4, 6, 0, 0, 0 }, /* 0x5B (01011011) */
            { 2, 3, 4, 6, 0, 0, 0, 0 }, /* 0x5C (01011100) */
            { 0, 2, 3, 4, 6, 0, 0, 0 }, /* 0x5D (01011101) */
            { 1, 2, 3, 4, 6, 0, 0, 0 }, /* 0x5E (01011110) */
            { 0, 1, 2, 3, 4, 6, 0, 0 }, /* 0x5F (01011111) */
            { 5, 6, 0, 0, 0, 0, 0, 0 }, /* 0x60 (01100000) */
            { 0, 5, 6, 0, 0, 0, 0, 0 }, /* 0x61 (01100001) */
            { 1, 5, 6, 0, 0, 0, 0, 0 }, /* 0x62 (01100010) */
            { 0, 1, 5, 6, 0, 0, 0, 0 }, /* 0x63 (01100011) */
            { 2, 5, 6, 0, 0, 0, 0, 0 }, /* 0x64 (01100100) */
            { 0, 2, 5, 6, 0, 0, 0, 0 }, /* 0x65 (01100101) */
            { 1, 2, 5, 6, 0, 0, 0, 0 }, /* 0x66 (01100110) */
            { 0, 1, 2, 5, 6, 0, 0, 0 }, /* 0x67 (01100111) */
            { 3, 5, 6, 0, 0, 0, 0, 0 }, /* 0x68 (01101000) */
            { 0, 3, 5, 6, 0, 0, 0, 0 }, /* 0x69 (01101001) */
            { 1, 3, 5, 6, 0, 0, 0, 0 }, /* 0x6A (01101010) */
            { 0, 1, 3, 5, 6, 0, 0, 0 }, /* 0x6B (01101011) */
            { 2, 3, 5, 6, 0, 0, 0, 0 }, /* 0x6C (01101100) */
            { 0, 2, 3, 5, 6, 0, 0, 0 }, /* 0x6D (01101101) */
            { 1, 2, 3, 5, 6, 0, 0, 0 }, /* 0x6E (01101110) */
            { 0, 1, 2, 3, 5, 6, 0, 0 }, /* 0x6F (01101111) */
            { 4, 5, 6, 0, 0, 0, 0, 0 }, /* 0x70 (01110000) */
            { 0, 4, 5, 6, 0, 0, 0, 0 }, /* 0x71 (01110001) */
            { 1, 4, 5, 6, 0, 0, 0, 0 }, /* 0x72 (01110010) */
            { 0, 1, 4, 5, 6, 0, 0, 0 }, /* 0x73 (01110011) */
            { 2, 4, 5, 6, 0, 0, 0, 0 }, /* 0x74 (01110100) */
            { 0, 2, 4, 5, 6, 0, 0, 0 }, /* 0x75 (01110101) */
            { 1, 2, 4, 5, 6, 0, 0, 0 }, /* 0x76 (01110110) */
            { 0, 1, 2, 4, 5, 6, 0, 0 }, /* 0x77 (01110111) */
            { 3, 4, 5, 6, 0, 0, 0, 0 }, /* 0x78 (01111000) */
            { 0, 3, 4, 5, 6, 0, 0, 0 }, /* 0x79 (01111001) */
            { 1, 3, 4, 5, 6, 0, 0, 0 }, /* 0x7A (01111010) */
            { 0, 1, 3, 4, 5, 6, 0, 0 }, /* 0x7B (01111011) */
            { 2, 3, 4, 5, 6, 0, 0, 0 }, /* 0x7C (01111100) */
            { 0, 2, 3, 4, 5, 6, 0, 0 }, /* 0x7D (01111101) */
            { 1, 2, 3, 4, 5, 6, 0, 0 }, /* 0x7E (01111110) */
            { 0, 1, 2, 3, 4, 5, 6, 0 }, /* 0x7F (01111111) */
            { 7, 0, 0, 0, 0, 0, 0, 0 }, /* 0x80 (10000000) */
            { 0, 7, 0, 0, 0, 0, 0, 0 }, /* 0x81 (10000001) */
            { 1, 7, 0, 0, 0, 0, 0, 0 }, /* 0x82 (10000010) */
            { 0, 1, 7, 0, 0, 0, 0, 0 }, /* 0x83 (10000011) */
            { 2, 7, 0, 0, 0, 0, 0, 0 }, /* 0x84 (10000100) */
            { 0, 2, 7, 0, 0, 0, 0, 0 }, /* 0x85 (10000101) */
            { 1, 2, 7, 0, 0, 0, 0, 0 }, /* 0x86 (10000110) */
            { 0, 1, 2, 7, 0, 0, 0, 0 }, /* 0x87 (10000111) */
            { 3, 7, 0, 0, 0, 0, 0, 0 }, /* 0x88 (10001000) */
            { 0, 3, 7, 0, 0, 0, 0, 0 }, /* 0x89 (10001001) */
            { 1, 3, 7, 0, 0, 0, 0, 0 }, /* 0x8A (10001010) */
            { 0, 1, 3, 7, 0, 0, 0, 0 }, /* 0x8B (10001011) */
            { 2, 3, 7, 0, 0, 0, 0, 0 }, /* 0x8C (10001100) */
            { 0, 2, 3, 7, 0, 0, 0, 0 }, /* 0x8D (10001101) */
            { 1, 2, 3, 7, 0, 0, 0, 0 }, /* 0x8E (10001110) */
            { 0, 1, 2, 3, 7, 0, 0, 0 }, /* 0x8F (10001111) */
            { 4, 7, 0, 0, 0, 0, 0, 0 }, /* 0x90 (10010000) */
            { 0, 4, 7, 0, 0, 0, 0, 0 }, /* 0x91 (10010001) */
            { 1, 4, 7, 0, 0, 0, 0, 0 }, /* 0x92 (10010010) */
            { 0, 1, 4, 7, 0, 0, 0, 0 }, /* 0x93 (10010011) */
            { 2, 4, 7, 0, 0, 0, 0, 0 }, /* 0x94 (10010100) */
            { 0, 2, 4, 7, 0, 0, 0, 0 }, /* 0x95 (10010101) */
            { 1, 2, 4, 7, 0, 0, 0, 0 }, /* 0x96 (10010110) */
            { 0, 1, 2, 4, 7, 0, 0, 0 }, /* 0x97 (10010111) */
            { 3, 4, 7, 0, 0, 0, 0, 0 }, /* 0x98 (10011000) */
            { 0, 3, 4, 7, 0, 0, 0, 0 }, /* 0x99 (10011001) */
            { 1, 3, 4, 7, 0, 0, 0, 0 }, /* 0x9A (10011010) */
            { 0, 1, 3, 4, 7, 0, 0, 0 }, /* 0x9B (10011011) */
            { 2, 3, 4, 7, 0, 0, 0, 0 }, /* 0x9C (10011100) */
            { 0, 2, 3, 4, 7, 0, 0, 0 }, /* 0x9D (10011101) */
            { 1, 2, 3, 4, 7, 0, 0, 0 }, /* 0x9E (10011110) */
            { 0, 1, 2, 3, 4, 7, 0, 0 }, /* 0x9F (10011111) */
            { 5, 7, 0, 0, 0, 0, 0, 0 }, /* 0xA0 (10100000) */
            { 0, 5, 7, 0, 0, 0, 0, 0 }, /* 0xA1 (10100001) */
            { 1, 5, 7, 0, 0, 0, 0, 0 }, /* 0xA2 (10100010) */
            { 0, 1, 5, 7, 0, 0, 0, 0 }, /* 0xA3 (10100011) */
            { 2, 5, 7, 0, 0, 0, 0, 0 }, /* 0xA4 (10100100) */
            { 0, 2, 5, 7, 0, 0, 0, 0 }, /* 0xA5 (10100101) */
            { 1, 2, 5, 7, 0, 0, 0, 0 }, /* 0xA6 (10100110) */
            { 0, 1, 2, 5, 7, 0, 0, 0 }, /* 0xA7 (10100111) */
            { 3, 5, 7, 0, 0, 0, 0, 0 }, /* 0xA8 (10101000) */
            { 0, 3, 5, 7, 0, 0, 0, 0 }, /* 0xA9 (10101001) */
            { 1, 3, 5, 7, 0, 0, 0, 0 }, /* 0xAA (10101010) */
            { 0, 1, 3, 5, 7, 0, 0, 0 }, /* 0xAB (10101011) */
            { 2, 3, 5, 7, 0, 0, 0, 0 }, /* 0xAC (10101100) */
            { 0, 2, 3, 5, 7, 0, 0, 0 }, /* 0xAD (10101101) */
            { 1, 2, 3, 5, 7, 0, 0, 0 }, /* 0xAE (10101110) */
            { 0, 1, 2, 3, 5, 7, 0, 0 }, /* 0xAF (10101111) */
            { 4, 5, 7, 0, 0, 0, 0, 0 }, /* 0xB0 (10110000) */
            { 0, 4, 5, 7, 0, 0, 0, 0 }, /* 0xB1 (10110001) */
            { 1, 4, 5, 7, 0, 0, 0, 0 }, /* 0xB2 (10110010) */
            { 0, 1, 4, 5, 7, 0, 0, 0 }, /* 0xB3 (10110011) */
            { 2, 4, 5, 7, 0, 0, 0, 0 }, /* 0xB4 (10110100) */
            { 0, 2, 4, 5, 7, 0, 0, 0 }, /* 0xB5 (10110101) */
            { 1, 2, 4, 5, 7, 0, 0, 0 }, /* 0xB6 (10110110) */
            { 0, 1, 2, 4, 5, 7, 0, 0 }, /* 0xB7 (10110111) */
            { 3, 4, 5, 7, 0, 0, 0, 0 }, /* 0xB8 (10111000) */
            { 0, 3, 4, 5, 7, 0, 0, 0 }, /* 0xB9 (10111001) */
            { 1, 3, 4, 5, 7, 0, 0, 0 }, /* 0xBA (10111010) */
            { 0, 1, 3, 4, 5, 7, 0, 0 }, /* 0xBB (10111011) */
            { 2, 3, 4, 5, 7, 0, 0, 0 }, /* 0xBC (10111100) */
            { 0, 2, 3, 4, 5, 7, 0, 0 }, /* 0xBD (10111101) */
            { 1, 2, 3, 4, 5, 7, 0, 0 }, /* 0xBE (10111110) */
            { 0, 1, 2, 3, 4, 5, 7, 0 }, /* 0xBF (10111111) */
            { 6, 7, 0, 0, 0, 0, 0, 0 }, /* 0xC0 (11000000) */
            { 0, 6, 7, 0, 0, 0, 0, 0 }, /* 0xC1 (11000001) */
            { 1, 6, 7, 0, 0, 0, 0, 0 }, /* 0xC2 (11000010) */
            { 0, 1, 6, 7, 0, 0, 0, 0 }, /* 0xC3 (11000011) */
            { 2, 6, 7, 0, 0, 0, 0, 0 }, /* 0xC4 (11000100) */
            { 0, 2, 6, 7, 0, 0, 0, 0 }, /* 0xC5 (11000101) */
            { 1, 2, 6, 7, 0, 0, 0, 0 }, /* 0xC6 (11000110) */
            { 0, 1, 2, 6, 7, 0, 0, 0 }, /* 0xC7 (11000111) */
            { 3, 6, 7, 0, 0, 0, 0, 0 }, /* 0xC8 (11001000) */
            { 0, 3, 6, 7, 0, 0, 0, 0 }, /* 0xC9 (11001001) */
            { 1, 3, 6, 7, 0, 0, 0, 0 }, /* 0xCA (11001010) */
            { 0, 1, 3, 6, 7, 0, 0, 0 }, /* 0xCB (11001011) */
            { 2, 3, 6, 7, 0, 0, 0, 0 }, /* 0xCC (11001100) */
            { 0, 2, 3, 6, 7, 0, 0, 0 }, /* 0xCD (11001101) */
            { 1, 2, 3, 6, 7, 0, 0, 0 }, /* 0xCE (11001110) */
            { 0, 1, 2, 3, 6, 7, 0, 0 }, /* 0xCF (11001111) */
            { 4, 6, 7, 0, 0, 0, 0, 0 }, /* 0xD0 (11010000) */
            { 0, 4, 6, 7, 0, 0, 0, 0 }, /* 0xD1 (11010001) */
            { 1, 4, 6, 7, 0, 0, 0, 0 }, /* 0xD2 (11010010) */
            { 0, 1, 4, 6, 7, 0, 0, 0 }, /* 0xD3 (11010011) */
            { 2, 4, 6, 7, 0, 0, 0, 0 }, /* 0xD4 (11010100) */
            { 0, 2, 4, 6, 7, 0, 0, 0 }, /* 0xD5 (11010101) */
            { 1, 2, 4, 6, 7, 0, 0, 0 }, /* 0xD6 (11010110) */
            { 0, 1, 2, 4, 6, 7, 0, 0 }, /* 0xD7 (11010111) */
            { 3, 4, 6, 7, 0, 0, 0, 0 }, /* 0xD8 (11011000) */
            { 0, 3, 4, 6, 7, 0, 0, 0 }, /* 0xD9 (11011001) */
            { 1, 3, 4, 6, 7, 0, 0, 0 }, /* 0xDA (11011010) */
            { 0, 1, 3, 4, 6, 7, 0, 0 }, /* 0xDB (11011011) */
            { 2, 3, 4, 6, 7, 0, 0, 0 }, /* 0xDC (11011100) */
            { 0, 2, 3, 4, 6, 7, 0, 0 }, /* 0xDD (11011101) */
            { 1, 2, 3, 4, 6, 7, 0, 0 }, /* 0xDE (11011110) */
            { 0, 1, 2, 3, 4, 6, 7, 0 }, /* 0xDF (11011111) */
            { 5, 6, 7, 0, 0, 0, 0, 0 }, /* 0xE0 (11100000) */
            { 0, 5, 6, 7, 0, 0, 0, 0 }, /* 0xE1 (11100001) */
            { 2, 5, 6, 7, 0, 0, 0, 0 }, /* 0xE2 (11100010) */
            { 0, 1, 5, 6, 7, 0, 0, 0 }, /* 0xE3 (11100011) */
            { 2, 5, 6, 7, 0, 0, 0, 0 }, /* 0xE4 (11100100) */
            { 0, 2, 5, 6, 7, 0, 0, 0 }, /* 0xE5 (11100101) */
            { 1, 2, 5, 6, 7, 0, 0, 0 }, /* 0xE6 (11100110) */
            { 0, 1, 2, 5, 6, 7, 0, 0 }, /* 0xE7 (11100111) */
            { 3, 5, 6, 7, 0, 0, 0, 0 }, /* 0xE8 (11101000) */
            { 0, 3, 5, 6, 7, 0, 0, 0 }, /* 0xE9 (11101001) */
            { 1, 3, 5, 6, 7, 0, 0, 0 }, /* 0xEA (11101010) */
            { 0, 1, 3, 5, 6, 7, 0, 0 }, /* 0xEB (11101011) */
            { 2, 3, 5, 6, 7, 0, 0, 0 }, /* 0xEC (11101100) */
            { 0, 2, 3, 5, 6, 7, 0, 0 }, /* 0xED (11101101) */
            { 1, 2, 3, 5, 6, 7, 0, 0 }, /* 0xEE (11101110) */
            { 0, 1, 2, 3, 5, 6, 7, 0 }, /* 0xEF (11101111) */
            { 4, 5, 6, 7, 0, 0, 0, 0 }, /* 0xF0 (11110000) */
            { 0, 4, 5, 6, 7, 0, 0, 0 }, /* 0xF1 (11110001) */
            { 1, 4, 5, 6, 7, 0, 0, 0 }, /* 0xF2 (11110010) */
            { 0, 1, 4, 5, 6, 7, 0, 0 }, /* 0xF3 (11110011) */
            { 2, 4, 5, 6, 7, 0, 0, 0 }, /* 0xF4 (11110100) */
            { 0, 2, 4, 5, 6, 7, 0, 0 }, /* 0xF5 (11110101) */
            { 1, 2, 4, 5, 6, 7, 0, 0 }, /* 0xF6 (11110110) */
            { 0, 1, 2, 4, 5, 6, 7, 0 }, /* 0xF7 (11110111) */
            { 3, 4, 5, 6, 7, 0, 0, 0 }, /* 0xF8 (11111000) */
            { 0, 3, 4, 5, 6, 7, 0, 0 }, /* 0xF9 (11111001) */
            { 0, 3, 4, 5, 6, 7, 0, 0 }, /* 0xFA (11111010) */
            { 0, 1, 3, 4, 5, 6, 7, 0 }, /* 0xFB (11111011) */
            { 2, 3, 4, 5, 6, 7, 0, 0 }, /* 0xFC (11111100) */
            { 0, 2, 3, 4, 5, 6, 7, 0 }, /* 0xFD (11111101) */
            { 1, 2, 3, 4, 5, 6, 7, 0 }, /* 0xFE (11111110) */
            { 0, 1, 2, 3, 4, 5, 6, 7 }  /* 0xFF (11111111) */
        };
        static const uint64_t * get_positions(uint64_t in){
            return lookup_table[in];
        }
    };
    // <= -std=c++14: we have to provide a definition for any static constexpr data members that are odr-use;
    //                otherwise undefined reference error
    constexpr uint64_t bitmap_lookup_tables<8, uint64_t>::lookup_table[][8];

}

#endif //MORPHSTORE_CORE_MORPHING_INTERMEDIATES_BITMAP_H