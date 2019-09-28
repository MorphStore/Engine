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
 * @file vbp.h
 * @brief The vertical bit-packed layout.
 * 
 * This header only includes the actual needed headers.
 */

#ifndef MORPHSTORE_CORE_MORPHING_VBP_H
#define MORPHSTORE_CORE_MORPHING_VBP_H

// Hand-written code for the format and for random access.
#include <core/morphing/vbp_format_rndaccess.h>

// Generated code of the routines for (de)compression and decompression with
// processing.
#include <core/morphing/vbp_routines.h>

#endif /* MORPHSTORE_CORE_MORPHING_VBP_H */