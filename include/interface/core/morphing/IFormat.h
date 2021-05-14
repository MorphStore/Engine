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


#ifndef MORPHSTORE_FORMAT_INTERFACES_H
#define MORPHSTORE_FORMAT_INTERFACES_H

#include <stdlibs>

namespace morphstore {
    /// Forward declaration of class format in <core/morphing/format.h>
    class format;
    
    /**
     * Template argument must be derived from class format.
     * @tparam TFormat A class dreived from class format.
     */
    #ifdef USE_CPP20_CONCEPTS
        template<class TFormat>
        concept IFormat = std::is_base_of<format, TFormat>::value;
    #else
        #define IFormat typename
    #endif
    
    
    
//    /// Base class for Format Interface.
//    class FormatBase {
//        //
//    };
//    template<class TFormat>
//    concept IFormat = std::is_base_of<FormatBase, TFormat>::value;

} // namespace

#endif //MORPHSTORE_FORMAT_INTERFACES_H
