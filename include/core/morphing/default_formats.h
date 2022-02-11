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
 * @file default_formats.h
 * @brief Macros for the default template specializations of formats whose
 * template parameters depend on the vector extension.
 * 
 * @todo Maybe this can be done using templates, but there could be problems
 * with "using" and partial template specialization.
 */

#ifndef MORPHSTORE_CORE_MORPHING_DEFAULT_FORMATS_H
#define MORPHSTORE_CORE_MORPHING_DEFAULT_FORMATS_H

#include <core/utils/math.h>
#include <core/utils/preprocessor.h>

namespace morphstore {
    
#define DEFAULT_STATIC_VBP_F(ve, bw) \
    SINGLE_ARG(static_vbp_f< \
            vbp_l<bw, ve::vector_helper_t::element_count::value> \
    >)
    
#define DEFAULT_STATIC_VBP_BYTE_F(ve, bw) \
    SINGLE_ARG(static_vbp_f< \
            vbp_l< \
                    round_up_to_multiple(bw, bitsPerByte), \
                    ve::vector_helper_t::element_count::value \
            > \
    >)

#define DEFAULT_DYNAMIC_VBP_F(ve) \
    SINGLE_ARG(dynamic_vbp_f< \
            ve::vector_helper_t::size_bit::value, \
            ve::vector_helper_t::size_byte::value, \
            ve::vector_helper_t::element_count::value \
    >)

#define DEFAULT_GROUP_SIMPLE_F(ve) \
    SINGLE_ARG(group_simple_f< \
            ve::vector_helper_t::element_count::value, \
            typename ve::base_t, \
            ve::vector_helper_t::size_byte::value \
    >)

#define DEFAULT_DELTA_DYNAMIC_VBP_F(ve) \
    SINGLE_ARG(delta_f< \
            1024, \
            ve::vector_helper_t::element_count::value, \
            dynamic_vbp_f< \
                    ve::vector_helper_t::size_bit::value, \
                    ve::vector_helper_t::size_byte::value, \
                    ve::vector_helper_t::element_count::value \
            > \
    >)

#define DEFAULT_FOR_DYNAMIC_VBP_F(ve) \
    SINGLE_ARG(for_f< \
            1024, \
            ve::vector_helper_t::element_count::value, \
            dynamic_vbp_f< \
                    ve::vector_helper_t::size_bit::value, \
                    ve::vector_helper_t::size_byte::value, \
                    ve::vector_helper_t::element_count::value \
            > \
    >)
    
}

#endif //MORPHSTORE_CORE_MORPHING_DEFAULT_FORMATS_H
