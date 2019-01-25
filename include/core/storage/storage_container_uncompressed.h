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
 * @file storage_container_uncompressed.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_STORAGE_STORAGE_CONTAINER_UNCOMPRESSED_H
#define MORPHSTORE_CORE_STORAGE_STORAGE_CONTAINER_UNCOMPRESSED_H

#include "storage_container.h"
#include "storage_helper.h"
#include "../utils/logger.h"


namespace morphstore { namespace storage {

template< typename T >
class storage_container_uncompressed : public abstract_storage_container< T > {
   public:
      storage_container_uncompressed( storage_container_meta_data< T > && p_MetaData, T const * const p_Data ):
         abstract_storage_container< T >{ std::move( p_MetaData ), p_Data } {
         debug( "Uncompressed Storage Container - ctor( storage_container_meta_data &&, Data =", p_Data, ")" );
      }
};


}}

#endif //MORPHSTORE_CORE_STORAGE_STORAGE_CONTAINER_UNCOMPRESSED_H
