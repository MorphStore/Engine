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
 * @author Patrick Damme
 * @todo TODOS?
 */

#include "column_helper.h"

#include "../morphing/format.h"

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_H


namespace morphstore {

enum class storage_persistence_type {
   PERPETUAL,
   EPHEMERAL
};

template< class F >
class column {
   static_assert(
      std::is_base_of< format, F >::value,
      "column: template parameter F must be a subclass of format"
   );
    
   public:
      // Creates an emphemeral column. Intended for intermediate results.
      column( size_t p_SizeAllocatedByte ) : column(
         storage_persistence_type::EPHEMERAL,
         p_SizeAllocatedByte
      ) {
         //
      };
         
      column( column< F > const & ) = delete;
      column( column< F > && ) = delete;
      column< F > & operator= ( column< F > const & ) = delete;
      column< F > & operator= ( column< F > && ) = delete;

      virtual ~column( ) {
//         debug( "Uncompressed Storage Container - dtor( ): Freeing", m_Data );
//         free( const_cast< void * >( static_cast< void const * const >( m_Data ) ) );
      }
      
   private:
      column_meta_data m_MetaData;
      voidptr_t m_Data;
      
      column(
         storage_persistence_type p_PersistenceType,
         size_t p_SizeAllocatedByte
      ) :
         m_MetaData{ 0, 0, p_SizeAllocatedByte },
         m_Data{
#ifndef MSV_NO_SELFMANAGED_MEMORY
            ( p_PersistenceType == storage_persistence_type::PERPETUAL )
            ? general_memory_manager::get_instance( ).allocate( p_SizeAllocatedByte )
            : malloc( p_SizeAllocatedByte )
#else
            malloc( p_SizeAllocatedByte );
#endif
         }
      {
         //
      }

   public:
      inline voidptr_t get_data( void ) const {
         return m_Data;
      }
      inline size_t get_count_values( void ) const {
         return m_MetaData.m_CountLogicalValues;
      }
      inline void set_count_values( size_t p_CountValues ) {
         m_MetaData.m_CountLogicalValues = p_CountValues;
      }
      inline size_t get_size_used_byte( void ) const {
         return m_MetaData.m_SizeUsedByte;
      }
      inline void set_size_used_byte( size_t p_SizeUsedByte ) {
         m_MetaData.m_SizeUsedByte = p_SizeUsedByte;
      }
      inline void set_meta_data( size_t p_CountValues, size_t p_SizeUsedByte )  {
         m_MetaData.m_CountLogicalValues = p_CountValues;
         m_MetaData.m_SizeUsedByte = p_SizeUsedByte;
      }
      
      // Creates a perpetual column. Intended for base data.
      static column< F > * create_perpetual_column( size_t p_SizeAllocByte ) {
         return new
            (
               general_memory_manager::get_instance( ).allocate(
                  sizeof( column< F > )
               )
            )
            column(
               storage_persistence_type::PERPETUAL,
               p_SizeAllocByte
            );
      }
};


}
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_H
