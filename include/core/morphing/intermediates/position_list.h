

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
 * @file position_list.h
 * @brief Position-List (PL) IR format with morphing-operators and other facilities.
 */

#ifndef MORPHSTORE_CORE_MORPHING_INTERMEDIATES_POSITION_LIST_H
#define MORPHSTORE_CORE_MORPHING_INTERMEDIATES_POSITION_LIST_H

#include <core/morphing/format.h>
#include <core/morphing/intermediates/representation.h>
#include <core/morphing/morph_batch.h>
#include <core/morphing/morph.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {

    using namespace vectorlib;

    /**
    * @brief Position-list (or row-id list) representation/format with some inner-format.
    *
    *        An IR-type consists of an inner-format (e.g. uncompr_f, rle_f, etc.) and its logical representation (BM, PL).
    *        Generally: IR<format> , e.g. position_list<rle_f> (run-length-encoded position-list)
    *
    *        By default, the inner-format is uncompr_f so that we can just write e.g. 'position_list' instead
    *        of 'position_list<uncompr_f>' (>= c++17). Otherwise (i.e. < c++17), we have to write 'position_list<>'.
    */
    template<class inner_format_t = uncompr_f>
    struct position_list_f : public intermediate_representation, public format {
        using t_inner_f = inner_format_t;

        static_assert(
                std::is_base_of<representation, t_inner_f>::value,
                "position_list_f: the template parameter t_inner_f must be a subclass of representation"
        );

        static size_t get_size_max_byte(size_t p_CountValues) {
            return t_inner_f::get_size_max_byte(p_CountValues);
        }

        static const size_t m_BlockSize = t_inner_f::m_BlockSize;

        static const intermediate_type ir_type = {intermediate_type::position_list};
    };

    // ------------------------------------------------------------------------
    // PL - Compression
    // ------------------------------------------------------------------------

    // ------------------------------ batch-level ------------------------------
    template<class t_vector_extension, class inner_format>
    struct morph_batch_t<t_vector_extension, position_list_f<inner_format>, position_list_f<uncompr_f> >{
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
    struct morph_t<t_vector_extension, position_list_f<inner_format>, position_list_f<uncompr_f> >{
        using dest_format = inner_format;

        static
        const column< position_list_f<dest_format> > *
        apply(
                const column< position_list_f<uncompr_f> > * inCol
        ) {
            // return IR-column: cast from inner_format -> position_list_f<inner_format>
            return
                    reinterpret_cast< const column< position_list_f<dest_format> > * >
                    (
                            morph<t_vector_extension, dest_format, uncompr_f>(
                                    // cast inCol to uncompr_f (from position_list_f<uncompr_f>)
                                    reinterpret_cast<const column<uncompr_f> *>(inCol)
                            )
                    );
        }
    };

    // ------------------------------------------------------------------------
    // PL - Decompression
    // ------------------------------------------------------------------------

    // ------------------------------ batch-level ------------------------------
    template<class t_vector_extension, class inner_format>
    struct morph_batch_t<t_vector_extension, position_list_f<uncompr_f>, position_list_f<inner_format> >{
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
    struct morph_t<t_vector_extension, position_list_f<uncompr_f>, position_list_f<inner_format> >{
        using src_format = inner_format;

        static
        const column< position_list_f<uncompr_f> > *
        apply(
                const column< position_list_f<src_format> > * inCol
        ) {
            // return IR-column: cast from inner_format -> position_list_f<inner_format>
            return
                    reinterpret_cast< const column< position_list_f<uncompr_f> > * >
                    (
                            morph<t_vector_extension, uncompr_f, src_format>(
                                    // cast inCol to uncompr_f (from position_list_f<uncompr_f>)
                                    reinterpret_cast<const column<src_format> *>(inCol)
                            )
                    );
        }
    };

    // ------------------------------------------------------------------------
    // Morph: position_list_f<uncompr_f> --> uncompr_f & vice versa
    // ------------------------------------------------------------------------

    // Need these additional template specialization for that case, otherwise 'error: use of deleted function ...'
    // in morph_batch-interface: just a wrapper to morph_batch<ve, uncompr_f, uncompr_f>
    template<class t_vector_extension>
    struct morph_batch_t<t_vector_extension, uncompr_f, position_list_f<uncompr_f> >{
        static void apply(
                const uint8_t * & p_In8,
                uint8_t * & p_Out8,
                size_t p_CountLog
        ) {
            morph_batch<t_vector_extension, uncompr_f, uncompr_f>(p_In8, p_Out8, p_CountLog);
        }
    };

    template<class t_vector_extension>
    struct morph_batch_t<t_vector_extension, position_list_f<uncompr_f>, uncompr_f >{
        static void apply(
                const uint8_t * & p_In8,
                uint8_t * & p_Out8,
                size_t p_CountLog
        ) {
            morph_batch<t_vector_extension, uncompr_f, uncompr_f>(p_In8, p_Out8, p_CountLog);
        }
    };

}

#endif //MORPHSTORE_CORE_MORPHING_INTERMEDIATES_POSITION_LIST_H