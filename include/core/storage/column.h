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

#include "column_helper.h"

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_H


namespace morphstore { namespace storage {

enum class storage_persistence_type {
   PERPETUAL,
   EPHEMERAL
};

template< typename T >
class column {
   public:


      column(
         storage_persistence_type p_PersistenceType,
         size_t p_SizeAllocatedByte
         ) :
         m_MetaData{ 0, 0, p_SizeAllocatedByte },
         m_Data{
#ifndef MSV_NO_SELFMANAGED_MEMORY
            ( p_PersistenceType == storage_persistence_type::PERPETUAL ) ?
            static_cast< T * >(
               morphstore::memory::general_memory_manager::get_instance().allocate_persist( p_SizeAllocatedByte )
               ) :
            static_cast< T * >( malloc( p_SizeAllocatedByte ) )
#else
            static_cast< T * >( malloc( p_SizeAllocatedByte ) );
#endif
         } { }


      column( column const & ) = delete;
      column( column && ) = delete;
      column & operator= ( column const & ) = delete;
      column & operator= ( column && ) = delete;

      virtual ~column( ) {
//         debug( "Uncompressed Storage Container - dtor( ): Freeing", m_Data );
//         free( const_cast< void * >( static_cast< void const * const >( m_Data ) ) );
      }
   private:
      storage_container_meta_data< T > m_MetaData;
      T const * m_Data;


   public:
      inline T const * data( void ) const {
         return m_Data;
      }
      inline T * data( void ) {
         return const_cast< T * >( m_Data );
      }
      inline size_t count_values( void ) const {
         return m_MetaData.m_CountLogicalValues;
      }
      inline void count_values( size_t p_CountValues ) {
         m_MetaData.m_CountLogicalValues = p_CountValues;
      }
      inline size_t size_used_byte( void ) const {
         return m_MetaData.m_SizeUsedByte;
      }
      inline void size_used_byte( size_t p_SizeUsedByte ) {
         m_MetaData.m_SizeUsedByte = p_SizeUsedByte;
      }
      inline void set_meta_data( size_t p_CountValues, size_t p_SizeUsedByte )  {
         m_MetaData.m_CountLogicalValues = p_CountValues;
         m_MetaData.m_SizeUsedByte = p_SizeUsedByte;
      }
};


} }
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_H
