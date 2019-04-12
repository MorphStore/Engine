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
 * @file morph.h
 * @brief The template-based interface of the morph-operator.
 */

#ifndef MORPHSTORE_CORE_MORPHING_MORPH_H
#define MORPHSTORE_CORE_MORPHING_MORPH_H

#include <core/storage/column.h>
#include <core/utils/processing_style.h>

namespace morphstore {

/**
 * @brief A struct wrapping the actual morph-operator.
 * 
 * This is necessary to enable partial template specialization, which is
 * required, since some compressed formats have their own template parameters.
 */
template<
        processing_style_t t_ps,
        class t_dst_f,
        class t_src_f
>
struct morph_t {
    /**
     * @brief Morph-operator. Changes the (compressed) format of the given
     * column from the source format `t_src_f` to the destination format
     * `t_dst_f` without logically changing the data.
     * 
     * This function is deleted by default, to guarantee that using this struct
     * with a format combination it is not specialized for causes a compiler
     * error, not a linker error.
     * 
     * @param inCol The data represented in the source format.
     * @return The same data represented in the destination format.
     */
    static
    const column<t_dst_f> *
    apply(const column<t_src_f> * inCol) = delete;
};

/**
 * @brief A template specialization of the morph-operator handling the case
 * when the source and the destination format are the same.
 * 
 * It merely returns the given column without doing any work.
 */
template<
        processing_style_t t_ps,
        class t_f
>
struct morph_t<t_ps, t_f, t_f> {
    static
    const column<t_f> *
    apply(const column<t_f> * inCol) {
        return inCol;
    };
};

/**
 * A convenience function wrapping the morph-operator.
 * 
 * Changes the (compressed) format of the given column from the source format
 * `t_src_f` to the destination format `t_dst_f` without logically changing the
 * data. 
 * 
 * @param inCol The data represented in the source format.
 * @return The same data represented in the destination format.
 */
template<
        processing_style_t t_ps,
        class t_dst_f,
        class t_src_f
>
const column<t_dst_f> * morph(const column<t_src_f> * inCol) {
    return morph_t<t_ps, t_dst_f, t_src_f>::apply(inCol);
}

}

#endif //MORPHSTORE_CORE_MORPHING_MORPH_H
