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
 * @file representation.h
 * @brief The base class of all intermediate representations + specific type traits helper.
 *
 */

#ifndef MORPHSTORE_CORE_MORPHING_REPRESENTATION_H
#define MORPHSTORE_CORE_MORPHING_REPRESENTATION_H

#include <type_traits>

namespace morphstore {

    /**
     * @brief Base class of all IR-types (BMs, PLs).
     *
     */
    struct intermediate_representation {};

    /**
     * @brief Little Workaround to get the "outer"-type of an IR. Currently we define an IR with its logical
     *        representation (bitmap_f, position_list_f) , i.e. outer-type, and its inner-format (uncompr_f, delta_f, for_f, etc.)
     *        as e.g. position_list_f<uncompr_f>.
     *        To get the inner_format, we simply can call position_list_f<uncompr_f>::t_inner_f to get 'uncompr_f'.
     *        But getting the outer_type (position_list_f or bitmap_f), type deduction on nested template classes
     *        turns out to be difficult, as every IR_type is bounded with an inner_format (design decision).
     *
     *        By defining an enum class with the IR-types as values, we can define additional type-traits that check
     *        the enum value of an IR and return true/false.
     *        For example, position_list_f<uncompr_f> has its ir_type (attribute) set to intermediate_type::position_list,
     *        and we can now check if this IR is of type 'position_list_f' or 'bitmap_f' using:
     *        is_position_list_t< position_list_f<uncompr_f> > => returns true
     *        is_bitmap_t< position_list_f<uncompr_f> > => returns false
     *
     *        Later on, we can use this evaluation in an operator's template arguments to instantiate an operator's code
     *        according to its underlying ir-type, i.e. bitmap / position-list.
     *        So far, we just checked if the template class of the IR is_base_of<intermediate_representation, ...> and
     *        for comparing the underlying data structure of 2 IRs with e.g. std::is_same< bitmap_f<uncompr_f>, bitmap_f<rle_f>>
     *        this evaluates to FALSE, although it should be TRUE.
     *        => Solution: see is_same_underlying_IR_t< ... >
     *        This is especially needed to "Develop a unified processing approach for position lists and bitmaps"
     *        (DA-GOAL 2), more specifically to call an operator's implementation according to its underlying intermediate
     *        data structure.
     *
     *        In the future, it would be nice to just have an "using t_outer_f = position_list_f / bitmap_f", but at
     *        this time, this is not possible. =>  workaround
     */
    enum class intermediate_type {
        position_list,
        bitmap
    };

    // check if a class is an IR, i.e. base_of_<intermediate_representation>
    template<class IR>
    struct is_intermediate_representation_t {
        static const bool value = std::is_base_of<intermediate_representation, IR>::value;
    };

    // check if an IR is a position-list
    template<
            class IR,
            typename std::enable_if_t< is_intermediate_representation_t<IR>::value,int> = 0>
    struct is_position_list_t {
        static const bool value = (IR::ir_type == intermediate_type::position_list);
    };

    // check if an IR is a bitmap
    template<
            class IR,
            typename std::enable_if_t< is_intermediate_representation_t<IR>::value,int> = 0>
    struct is_bitmap_t {
        static const bool value = (IR::ir_type == intermediate_type::bitmap);
    };

    // check if two IR's have the same underlying IR data structure, i.e. bitmap or position-list
    template<
            class IR_1,
            class IR_2,
            typename std::enable_if_t<
                    is_intermediate_representation_t<IR_1>::value &&
                    is_intermediate_representation_t<IR_2>::value
                    ,int> = 0>
    struct is_same_underlying_IR_t {
        static const bool value = (IR_1::ir_type == IR_2::ir_type);
    };
}

#endif //MORPHSTORE_CORE_MORPHING_REPRESENTATION_H