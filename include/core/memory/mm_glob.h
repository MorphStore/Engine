/**********************************************************************************************
 * Copyright (C) 2019 by Johannes Pietrzyk                                                    *
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
 * @file mm_glob.h
 *
 * @author Johannes Pietrzyk
 *
 * @todo What happens, if MSV_NO_SELFMANAGED_MEMORY is defined. ( take care of this in logger.h )
 */

#ifndef MORPHSTORE_CORE_MEMORY_MM_GLOB_H
#define MORPHSTORE_CORE_MEMORY_MM_GLOB_H

#ifndef MSV_NO_SELFMANAGED_MEMORY
#  ifndef MSV_MEMORY_MANAGER_ALIGNMENT_BYTE
#     include "../utils/helper.h"
#     define MSV_MEMORY_MANAGER_ALIGNMENT_BYTE 64_B
#  endif
#  define MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE (MSV_MEMORY_MANAGER_ALIGNMENT_BYTE-1)

#  include "morphstore_mm.h"
#endif





#endif //MORPHSTORE_CORE_MEMORY_MM_GLOB_H
