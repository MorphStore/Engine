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

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_H

#include <core/storage/column_helper.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/morph_batch.h>
#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>

#include <type_traits>

#include <cstring>

namespace morphstore {

enum class storage_persistence_type {
   globalScope,
   queryScope
};

// template-free base class
// use-case: graph formats can change their column format at run-time via `compress(Format f)`
class column_base {
   public: 
      virtual ~column_base() {}
      // todo: find a way to specify `inline`
      virtual voidptr_t get_data( void ) const = 0;
      virtual size_t get_count_values( void ) const = 0;
      virtual void set_count_values( size_t p_CountValues ) = 0;
      virtual size_t get_size_used_byte( void ) const = 0;
      virtual void set_size_used_byte( size_t p_SizeUsedByte ) = 0;
      virtual size_t get_size_compr_byte( void ) const = 0;
      virtual void set_size_compr_byte( size_t p_SizeComprByte ) = 0;
      virtual void set_meta_data( size_t p_CountValues, size_t p_SizeUsedByte, size_t p_SizeComprByte ) = 0;
      virtual void set_meta_data( size_t p_CountValues, size_t p_SizeUsedByte) = 0;

      virtual const voidptr_t get_data_uncompr_start() const = 0;
      virtual size_t get_count_values_uncompr() const = 0;
      virtual size_t get_count_values_compr() const = 0;
      // this is a template-method and cannot be defined here?
      //virtual bool prepare_for_random_access() const = 0;
};

template< class F >
class column : public column_base {
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
#ifdef MSV_NO_SELFMANAGED_MEMORY
         free( m_DataUnaligned );
#else
         if( m_PersistenceType == storage_persistence_type::globalScope ) {
            general_memory_manager::get_instance( ).deallocate( m_Data );
         }
#endif
      }
      
   private:
      column_meta_data m_MetaData;
#ifdef MSV_NO_SELFMANAGED_MEMORY
      void * m_DataUnaligned;
#endif
      voidptr_t m_Data;
      // @todo Actually, the persistence type only makes sense if we use our
      // own memory manager.
      storage_persistence_type m_PersistenceType;
      
      mutable bool m_IsPreparedForRndAccess;
      
      column(
         storage_persistence_type p_PersistenceType,
         size_t p_SizeAllocatedByte
      ) :
         m_MetaData{ 0, 0, 0, p_SizeAllocatedByte },
#ifdef MSV_NO_SELFMANAGED_MEMORY
         m_DataUnaligned{
            malloc( get_size_with_alignment_padding( p_SizeAllocatedByte ) )
         },
         m_Data{ create_aligned_ptr( m_DataUnaligned ) },
#else
         m_Data{
            ( p_PersistenceType == storage_persistence_type::globalScope )
            ? general_memory_manager::get_instance( ).allocate( p_SizeAllocatedByte )
            : malloc( p_SizeAllocatedByte )
         },
#endif
         m_PersistenceType{ p_PersistenceType },
         m_IsPreparedForRndAccess(false)
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
      
        // The following utility functions for working with the subdivision of
        // a column into a compressed main part and an uncompressed rest part
        // can definitely be simplified. However, they work and should not
        // matter w.r.t. performance anyway.
      
        /**
         * @brief Returns a pointer to the start of the uncompressed rest part
         * of the column.
         * 
         * If the column's format is uncompressed or there is no compressed
         * data in the column, then this will point to the start of the
         * column's data buffer.
         * 
         * If the column does not contain any uncompressed data, then the
         * returned pointer will point to the address where this data would
         * start if it existed.
         * 
         * @return A pointer to the start of the uncompressed rest part of the
         * column.
         */
        inline const voidptr_t get_data_uncompr_start() const {
            return voidptr_t(create_data_uncompr_start(
                    static_cast<uint8_t *>(m_Data) +
                    m_MetaData.m_SizeComprByte
            ));
        }
        
        /**
         * @brief Returns a pointer to the byte where the uncompressed rest
         * part of a column would start, given the end of the compressed main
         * part.
         * 
         * @param p_ComprEnd A pointer to the byte immediately behind the
         * compressed main part.
         * @return A pointer to the start of the uncompressed rest part.
         */
        inline static uint8_t * create_data_uncompr_start(uint8_t * p_ComprEnd) {
            if(std::is_same<F, uncompr_f>::value)
                return create_aligned_ptr(p_ComprEnd);
            else
                // Between the compressed main part and the uncompressed rest
                // part, we reserve enough space for two more compressed
                // blocks. This space is needed when the column is prepared for
                // random access in the project-operator. We need space for two
                // compressed blocks, since the size of the uncompressed rest
                // can exceed the size of one block, due to the way the
                // operators handle the scalar part of the uncompressed rest.
                return create_aligned_ptr(
                        p_ComprEnd + 2 * F::get_size_max_byte(F::m_BlockSize)
                );
        }
        
        /**
         * Returns the number of logical data elements in the uncompressed rest
         * part of the column.
         * 
         * @return The number of data elements in the uncompressed rest part of
         * the column.
         */
        inline size_t get_count_values_uncompr() const {
            return (m_MetaData.m_SizeComprByte == m_MetaData.m_SizeUsedByte)
                    ? 0
                    : convert_size<uint8_t, uint64_t>(
                            m_MetaData.m_SizeUsedByte - (
                                    static_cast<const uint8_t *>(get_data_uncompr_start()) -
                                    static_cast<const uint8_t *>(m_Data.m_Ptr)
                            )
                    );
        }
        
        /**
         * Returns the number of logical data elements in the compressed main
         * part of the column.
         * 
         * @return The number of data elements in the compressed main part of
         * the column.
         */
        inline size_t get_count_values_compr() const {
            return m_MetaData.m_CountLogicalValues - get_count_values_uncompr();
        }
        
        /**
         * @brief Prepares this column for random access.
         * 
         * To enable random access to any logical position *on compressed
         * data*, the uncompressed rest part is padded with zeros and appended
         * in compressed form to the compressed main part.
         * 
         * No action is taken if any of the following holds:
         * - this column has been prepared for random access before
         * - the format of this column is uncompressed
         * - the uncompressed rest part is empty
         * 
         * @return `true` if action was taken, `false` otherwise.
         */
        template<class t_vector_extension>
        bool prepare_for_random_access() const {
            if(m_IsPreparedForRndAccess || std::is_same<F, uncompr_f>::value)
                return false;
            
            const size_t countLogUncompr = get_count_values_uncompr();
            if(!countLogUncompr)
                return false;
            
            const uint8_t * rest8 = get_data_uncompr_start();
            const size_t countLogUncomprRoundUp =
                    round_up_to_multiple(countLogUncompr, F::m_BlockSize);
            memset(
                    const_cast<uint8_t *>(rest8) + convert_size<uint64_t, uint8_t>(countLogUncompr),
                    0,
                    convert_size<uint64_t, uint8_t>(
                            countLogUncomprRoundUp - countLogUncompr
                    )
            );
            uint8_t * comprEnd =
                    static_cast<uint8_t *>(m_Data) + m_MetaData.m_SizeComprByte;
            morph_batch<t_vector_extension, F, uncompr_f>(
                    rest8, comprEnd, countLogUncomprRoundUp
            );
            
            m_IsPreparedForRndAccess = true;
            return true;
        }
        
      // Creates a global scoped column. Intended for base data.
      static column< F > * create_global_column(size_t p_SizeAllocByte) {
         return new
            (
#ifdef MSV_NO_SELFMANAGED_MEMORY
               malloc( sizeof( column< F > ) )
#else
               general_memory_manager::get_instance( ).allocate(
                  sizeof( column< F > )
               )
#endif
            )
            column(
               storage_persistence_type::globalScope,
               p_SizeAllocByte
            );
      }
};
}
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_H
