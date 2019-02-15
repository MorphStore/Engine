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

#include "global/mm_hooks.h"

#include "management/allocators/perpetual_allocator.h"
#include "stl_wrapper/ostream.h"
#include "stl_wrapper/string.h"



#ifdef MSV_MEMORY_LEAK_CHECK
#  include "global/leak_detection.h"
#endif
#include "../utils/logger.h"

#include "../utils/helper_types.h"

#include "management/abstract_mm.h"
#include "management/utils/expand_helper.h"
#include "management/utils/memory_bin_handler.h"
#include "management/general_mm.h"
#include "management/query_mm.h"

#include "global/mm_stdlib.h"


#endif //MORPHSTORE_MORPHSTORE_MM_H