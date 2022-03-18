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
 * @file interfaces.h
 * @brief General interfaces and primary templates to transform an intermediate rep. to another intermediate rep.
 *        (transform_IR_t<> and transform_IR_batch<>)
 *
 */

#ifndef MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_INTERFACES_H
#define MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_INTERFACES_H

#include <core/morphing/intermediates/position_list.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/intermediates/representation.h>

#include <type_traits>

namespace morphstore {

    // ****************************************************************************
    // Column-level
    // ****************************************************************************

    // primary template
    template<
            class t_vector_extension,
            class t_IR_dst_f,
            class t_IR_src_f,
            typename std::enable_if_t<
                    // enable only if src & dest are IR-types, i.e. position_list_f<> and bitmap_f<>
                    is_intermediate_representation_t<t_IR_src_f>::value &&
                    is_intermediate_representation_t<t_IR_dst_f>::value
            , int> = 0
    >
    struct transform_IR_t {
        static const column <t_IR_dst_f> *
        apply(
                const column <t_IR_src_f> *inCol
        ) = delete;
    };

    // A convenience function wrapping the transform-IR-operator (column)
    template<
            class t_vector_extension,
            class t_IR_dst_f,
            class t_IR_src_f,
            typename std::enable_if_t<
                    // enable only if src & dest are IR-types, i.e. position_list_f<> and bitmap_f<>
                    is_intermediate_representation_t<t_IR_src_f>::value &&
                    is_intermediate_representation_t<t_IR_dst_f>::value, int> = 0
    >
    const column <t_IR_dst_f> * transform_IR(const column <t_IR_src_f> *inCol) {
        return transform_IR_t<t_vector_extension, t_IR_dst_f, t_IR_src_f>::apply(inCol);
    }

    /**
     * @brief A template specialization of the morph-operator handling the case
     *        when the source and the destination IR-types are the same and their
     *        inner-formats are uncompressed. (COLUMN-LEVEL)
     *
     *        We need to make this case explicit, since otherwise, the choice of the
     *        right partial template specialization is ambiguous for the compiler.
     */

    // bitmap
    template<class t_vector_extension>
    struct transform_IR_t<t_vector_extension, bitmap_f<>, bitmap_f<>> {
        static
        const column <bitmap_f<>> *
        apply(const column <bitmap_f<>> *inCol) {
            return inCol;
        };
    };

    // position-list
    template<class t_vector_extension>
    struct transform_IR_t<t_vector_extension, position_list_f<>, position_list_f<>> {
        static
        const column <position_list_f<>> *
        apply(const column <position_list_f<>> *inCol) {
            return inCol;
        };
    };

    // ****************************************************************************
    // Batch-level
    // ****************************************************************************

    /**
     * @brief Batch-level processing of intermediate transformation
     *
     * Important: @param startingPos
     *            Denotes the current (bit) position to have a starting point when transforming from
     *            a batch of BMs to PLs.
     */

    // primary template
    template<
            class t_vector_extension,
            class t_IR_dst_f,
            class t_IR_src_f,
            typename std::enable_if_t<
                    // enable only if src & dest are IR-types, i.e. position_list_f<> and bitmap_f<>
                    (is_intermediate_representation_t<t_IR_src_f>::value &&
                    is_intermediate_representation_t<t_IR_dst_f>::value)
                    ||
                    // special case to support uncompr_f, assuming position-list (support existing implementation) TODO: remove this later
                    (std::is_same<t_IR_src_f, uncompr_f>::value && std::is_same<t_IR_dst_f, uncompr_f>::value)
            , int> = 0
    >
    struct transform_IR_batch_t {
        static void apply(
                const uint8_t *&in8,
                uint8_t *&out8,
                size_t countLog,
                const uint64_t startingPos
        ) = delete;
    };

    // A convenience function wrapping the transform-IR-operator (batch)
    template<
            class t_vector_extension,
            class t_IR_dst_f,
            class t_IR_src_f,
            typename std::enable_if_t<
                    // enable only if src & dest are IR-types, i.e. position_list_f<> and bitmap_f<>
                    is_intermediate_representation_t<t_IR_src_f>::value &&
                    is_intermediate_representation_t<t_IR_dst_f>::value, int> = 0
    >
    void transform_IR_batch(
            const uint8_t *&in8,
            uint8_t *&out8,
            size_t countLog,
            const uint64_t startingPos
    ) {
        transform_IR_batch_t<t_vector_extension, t_IR_dst_f, t_IR_src_f>::apply(
                in8, out8, countLog, startingPos
        );
    }

    // Partial specialization for transforming from bitmap_f<> to bitmap_f<>
    template<class t_vector_extension>
    struct transform_IR_batch_t<t_vector_extension, bitmap_f<>, bitmap_f<>> {
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                const uint64_t startingPos
        ) {
            // to satisfy compiler warning "unused parameter"
            (void) startingPos;
            const size_t sizeByte = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, sizeByte);
            in8 += sizeByte;
            out8 += sizeByte;
        };
    };

    // Partial specialization for transforming from position_list_f<> to position_list_f<>
    template<class t_vector_extension>
    struct transform_IR_batch_t<t_vector_extension, position_list_f<>, position_list_f<>> {
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                const uint64_t startingPos
        ) {
            // to satisfy compiler error: unused parameter
            (void) startingPos;

            const size_t sizeByte = convert_size<uint64_t, uint8_t>(countLog);
            memcpy(out8, in8, sizeByte);
            in8 += sizeByte;
            out8 += sizeByte;
        };
    };

    /**
     * @brief Special Partial specialization for transforming from uncompr_f<> to uncompr_f<>.
     *        This is used in write_iterator_IR.h to support existing implementation.
     *
     *        Note: When uncompr_f is used in this template, we assume that the data elements are postions
     *              in a position-list, i.e. uint64_t numbers.
     *
     */
    template<class t_vector_extension>
    struct transform_IR_batch_t<t_vector_extension, uncompr_f, uncompr_f> {
        static void apply(
                const uint8_t * & in8,
                uint8_t * & out8,
                size_t countLog,
                const uint64_t startingPos
        ) {
            transform_IR_batch_t<t_vector_extension, position_list_f<>, position_list_f<> >::apply(
                    in8, out8, countLog, startingPos
            );
        };
    };

}

#endif //MORPHSTORE_CORE_MORPHING_INTERMEDIATES_TRANSFORMATIONS_INTERFACES_H