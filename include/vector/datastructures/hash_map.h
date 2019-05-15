//
// Created by jpietrzyk on 10.05.19.
//

#ifndef MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H
#define MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H

#include <core/memory/mm_glob.h>
#include <core/utils/preprocessor.h>
#include <vector/general_vector.h>
#include <vector/complex/hash.h>

namespace vector {

   template<
      class VectorExtension,
      template<class> class HashFunction,
      float MaxLoadfactor
   >
   class hash_map_constant_size_linear_probing {
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)

      public:
         MSV_CXX_ATTRIBUTE_FORCE_INLINE
         static base_t


      public:
         hash_map_constant_size_linear_probing (size_t const p_Size) :
            m_size{ p_Size + p_Size * MaxLoadfactor },
            m_Keys{ ( KeyT * ) malloc( m_Size * sizeof( base_t ) ) },
            m_Values{ ( ValueT * ) malloc( m_Size * sizeof( base_t ) ) } {
         }

         ~hash_map_constant_size_linear_probing () {
            free( m_Keys );
            free( m_values );
         }
      private:
         size_t const m_Size;
         base_t  * m_Keys;
         base_t  * m_Values;
      public:
         base_t  * const get_key_data( void ) const {
            return m_Keys;
         }
         base_t * const get_value_data( void ) const {
            return m_Values;
         }
         size_t get_element_count( void ) const {
            return m_Size;
         }
   };



}

#endif //MORPHSTORE_VECTOR_DATASTRUCTURES_HASH_MAP_H