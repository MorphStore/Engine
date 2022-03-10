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
 * @file format_names.h
 * @brief A template-based mapping from format structs to string names. Useful
 * especially (and perhaps exclusively) in micro benchmarks.
 * 
 * There are situations where we cannot accomplish this mapping using macros.
 * 
 * Note that some of the mappings abstract from certain template parameters of
 * the formats and replace them by a placeholder. These abstractions might not
 * be wanted in all cases, which is one of the reasons why we do not integrate
 * this mapping mechanism deeply into the formats themselves.
 * 
 * This header must be included /after/ the headers of all formats it shall
 * provide mappings for, since it depends on those headers' guards.
 */

#ifndef MORPHSTORE_CORE_MORPHING_FORMAT_NAMES_H
#define MORPHSTORE_CORE_MORPHING_FORMAT_NAMES_H

#include <core/utils/basic_types.h>

#include <string>


namespace morphstore {

    // All template-specializations of a format are mapped to a name, which may
    // or may not contain the values of the template parameters.
    
    template<class t_format>
    // Ideally, formatName should only be usable with formats it has explicitly
    // been specialized for (otherwise compiler errors should be thrown), but:
    // This does not compile.
    // std::string formatName = delete;
    // This results in empty string for formats w/o a specialization.
    //std::string formatName;
    // This results in this string for formats w/o a specialization.
    std::string formatName = "(unknown format)";

#ifdef MORPHSTORE_CORE_MORPHING_DYNAMIC_VBP_H
    template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, unsigned t_Step>
    std::string formatName<
            dynamic_vbp_f<t_BlockSizeLog, t_PageSizeBlocks, t_Step>
    > = "dynamic_vbp_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_PageSizeBlocks) + ", " + std::to_string(t_Step) + ">";
#endif

#ifdef MORPHSTORE_CORE_MORPHING_K_WISE_NS_H
    template<size_t t_BlockSizeLog>
    std::string formatName<k_wise_ns_f<t_BlockSizeLog>> = "k_wise_ns_f<" + std::to_string(t_BlockSizeLog) + ">";
#endif

#ifdef MORPHSTORE_CORE_MORPHING_STATIC_VBP_H
    template<unsigned t_Bw, unsigned t_Step>
    std::string formatName<
            static_vbp_f<vbp_l<t_Bw, t_Step> >
    > = "static_vbp_f<vbp_l<bw, " + std::to_string(t_Step) + "> >";
#endif

#ifdef MORPHSTORE_CORE_MORPHING_DELTA_H
    template<size_t t_BlockSizeLog, unsigned t_Step, class t_inner_f>
    std::string formatName<
            delta_f<t_BlockSizeLog, t_Step, t_inner_f>
    > = "delta_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_Step) + ", " + formatName<t_inner_f> + ">";
#endif

#ifdef MORPHSTORE_CORE_MORPHING_FOR_H
    template<size_t t_BlockSizeLog, size_t t_PageSizeBlocks, class t_inner_f>
    std::string formatName<
            for_f<t_BlockSizeLog, t_PageSizeBlocks, t_inner_f>
    > = "for_f<" + std::to_string(t_BlockSizeLog) + ", " + std::to_string(t_PageSizeBlocks) + ", " + formatName<t_inner_f> + ">";
#endif

    // @todo Currently, uncompr_f is defined in format.h, but it should have
    // its own header.
#ifdef MORPHSTORE_CORE_MORPHING_FORMAT_H
    template<>
    std::string formatName<uncompr_f> = "uncompr_f";
#endif
    
}

#endif //MORPHSTORE_CORE_MORPHING_FORMAT_NAMES_H
