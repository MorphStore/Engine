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
 * @file morphstore_mm.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_MORPHSTORE_MM_H
#define MORPHSTORE_MORPHSTORE_MM_H

#define USE_MMAP_MM

#include <core/utils/helper.h>
#include <core/memory/management/utils/expand_helper.h>

#include <core/memory/global/mm_hooks.h>
#include <core/memory/management/allocators/global_scope_allocator.h>
#include <core/memory/stl_wrapper/ostream.h>

#include <core/memory/stl_wrapper/string.h>
#ifdef MSV_MEMORY_LEAK_CHECK
#  include <core/memory/global/leak_detection.h>
#endif

#include <core/utils/logger.h>

#ifdef USE_MMAP_MM
#include <core/memory/management/abstract_mm.h>
#include <core/memory/global/mm_override.h>
#endif

#include <core/utils/helper_types.h>
#include <core/memory/management/abstract_mm.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/memory/management/utils/memory_bin_handler.h>
#include <core/memory/management/general_mm.h>
#include <core/memory/management/query_mm.h>
#include <core/memory/management/mmap_mm.h>
#include <core/memory/management/paged_mm.h>

#include <core/memory/global/mm_stdlib.h>


#endif //MORPHSTORE_MORPHSTORE_MM_H
