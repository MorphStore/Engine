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
 * @file column.h
 * @brief Brief description
 * @todo Which headers should be included to use the memory manager here?
 */

#include <core/storage/column_helper.h>
#ifdef MSV_NO_SELFMANAGED_MEMORY
#include <core/memory/management/utils/alignment_helper.h>
#endif
#include <core/morphing/format.h>
#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_H


namespace morphstore {

enum class storage_persistence_type {
   globalScope,
   queryScope
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
         storage_persistence_type::queryScope,
         p_SizeAllocatedByte
      ) {
         //
      };
         
      column( column< F > const & ) = delete;
      column( column< F > && ) = delete;
      column< F > & operator= ( column< F > const & ) = delete;
      column< F > & operator= ( column< F > && ) = delete;

      virtual ~column( ) {
//#ifdef MSV_NO_SELFMANAGED_MEMORY
         free( m_DataUnaligned );
/*#else
         if( m_PersistenceType == storage_persistence_type::globalScope ) {
            general_memory_manager::get_instance( ).deallocate( m_Data );
         }
#endif*/
      }
      
   private:
      column_meta_data m_MetaData;
//#ifdef MSV_NO_SELFMANAGED_MEMORY
      void * m_DataUnaligned;
//#endif
      voidptr_t m_Data;
      // @todo Actually, the persistence type only makes sense if we use our
      // own memory manager.
      storage_persistence_type m_PersistenceType;
      
      column(
         storage_persistence_type p_PersistenceType,
         size_t p_SizeAllocatedByte
      ) :
         m_MetaData{ 0, 0, 0, p_SizeAllocatedByte },
//#ifdef MSV_NO_SELFMANAGED_MEMORY
         m_DataUnaligned{
            malloc( get_size_with_alignment_padding( p_SizeAllocatedByte ) )
         },
         m_Data{ create_aligned_ptr( m_DataUnaligned ) },
/*#else
         m_Data{
            ( p_PersistenceType == storage_persistence_type::globalScope )
            ? general_memory_manager::get_instance( ).allocate( p_SizeAllocatedByte )
            : malloc( p_SizeAllocatedByte )
         },
#endif*/
         m_PersistenceType{ p_PersistenceType }
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
      inline size_t get_size_compr_byte( void ) const {
         return m_MetaData.m_SizeComprByte;
      }
      inline void set_size_compr_byte( size_t p_SizeComprByte ) {
         m_MetaData.m_SizeComprByte = p_SizeComprByte;
      }
      inline void set_meta_data(
         size_t p_CountValues,
         size_t p_SizeUsedByte,
         size_t p_SizeComprByte
      )  {
         m_MetaData.m_CountLogicalValues = p_CountValues;
         m_MetaData.m_SizeUsedByte = p_SizeUsedByte;
         m_MetaData.m_SizeComprByte = p_SizeComprByte;
      }
      inline void set_meta_data( size_t p_CountValues, size_t p_SizeUsedByte )  {
          set_meta_data(p_CountValues, p_SizeUsedByte, 0);
      }
      
      // Creates a global scoped column. Intended for base data.
      static column< F > * create_global_column(size_t p_SizeAllocByte) {
         return new
            (
//#ifdef MSV_NO_SELFMANAGED_MEMORY
               malloc( sizeof( column< F > ) )
/*#else
               general_memory_manager::get_instance( ).allocate(
                  sizeof( column< F > )
               )
#endif*/
            )
            column(
               storage_persistence_type::globalScope,
               p_SizeAllocByte
            );
      }
};


}
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_H
