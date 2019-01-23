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
 * @file storage_helper.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_STORAGE_STORAGE_HELPER_H
#define MORPHSTORE_CORE_STORAGE_STORAGE_HELPER_H

#include "../utils/types.h"

namespace morphstore { namespace storage {

template< typename T >
struct storage_container_meta_data {
   using data_type = T;
   constexpr size_t c_DataTypeBitwidth = sizeof( T ) * 8;
   size_t const m_CountLogicalValues;
   size_t const m_SizeByte;

   meta_data( size_t p_CountLogicalValues, size_t p_SizeByte ) :
      m_CountLogicalValues{ p_CountLogicalValues },
      m_SizeByte{ p_SizeByte }{ }
   meta_data( meta_data const & ) = delete;
   meta_data( meta_data && ) = default;
   meta_data & operator=( meta_data const & ) = delete;
   meta_data & operator=( meta_data && ) = default;
};


} }

#endif //MORPHSTORE_CORE_STORAGE_STORAGE_HELPER_H
