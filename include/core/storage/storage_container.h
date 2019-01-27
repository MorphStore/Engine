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
 * @file storage_container.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "storage_helper.h"

#ifndef MORPHSTORE_CORE_STORAGE_STORAGE_CONTAINER_H
#define MORPHSTORE_CORE_STORAGE_STORAGE_CONTAINER_H


namespace morphstore { namespace storage {

enum class storage_persistence_type {
   PERPETUAL,
   EPHEMERAL
};

template< typename T >
class abstract_storage_container {
   public:


      abstract_storage_container(
         storage_persistence_type p_PersistenceType,
         size_t p_CountLogicalValues,
         size_t p_SizeByte
         ) :
         m_MetaData{ p_CountLogicalValues, p_SizeByte },
         m_Data{
#ifndef MSV_NO_SELFMANAGED_MEMORY
            ( p_PersistenceType == storage_persistence_type::PERPETUAL ) ?
            static_cast< T * >( morphstore::memory::general_memory_manager::get_instance().allocate_persist( p_SizeByte ) ) :
            static_cast< T * >( malloc( p_SizeByte ) )
#else
            static_cast< T * >( malloc( p_SizeByte ) );
#endif
         } { }


      abstract_storage_container( abstract_storage_container const & ) = delete;
      abstract_storage_container( abstract_storage_container && ) = delete;
      abstract_storage_container & operator= ( abstract_storage_container const & ) = delete;
      abstract_storage_container & operator= ( abstract_storage_container && ) = delete;

      virtual ~abstract_storage_container( ) {
//         debug( "Uncompressed Storage Container - dtor( ): Freeing", m_Data );
//         free( const_cast< void * >( static_cast< void const * const >( m_Data ) ) );
      }
   protected:
      storage_container_meta_data< T > m_MetaData;
      T const * m_Data;


   public:
      inline T const * data( void ) const {
         return m_Data;
      }
      inline size_t count_values( void ) const {
         return m_MetaData.m_CountLogicalValues;
      }
      inline size_t size_byte( void ) const {
         return m_MetaData.m_SizeByte;
      }
};


} }
#endif //MORPHSTORE_CORE_STORAGE_STORAGE_CONTAINER_H
