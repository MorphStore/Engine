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
 * @file column_helper.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_HELPER_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_HELPER_H

#include "../utils/basic_types.h"
#include "../utils/logger.h"

namespace morphstore {

struct column_meta_data {
   size_t m_CountLogicalValues;
   size_t m_SizeUsedByte;
   // TODO make this const again
   size_t /*const*/ m_SizeAllocByte;

   column_meta_data( size_t p_CountLogicalValues, size_t p_SizeUsedByte, size_t p_SizeAllocByte ) :
      m_CountLogicalValues{ p_CountLogicalValues },
      m_SizeUsedByte{ p_SizeUsedByte },
      m_SizeAllocByte{ p_SizeAllocByte }{
      trace(
         "Column Meta Data - ctor( |Logical Values| =", p_CountLogicalValues,
         ", |Data| =", p_SizeUsedByte, "Byte",
         ", Allocated Size = ", p_SizeAllocByte, " Bytes ).");
   }
      
   column_meta_data( column_meta_data const & ) = delete;
   column_meta_data( column_meta_data && ) = default;
   column_meta_data & operator=( column_meta_data const & ) = delete;
   column_meta_data & operator=( column_meta_data && that) {
       m_CountLogicalValues = that.m_CountLogicalValues;
       m_SizeUsedByte = that.m_SizeUsedByte;
       m_SizeAllocByte = that.m_SizeAllocByte;
       return *this;
   }
};


}

#endif //MORPHSTORE_CORE_STORAGE_COLUMN_HELPER_H
