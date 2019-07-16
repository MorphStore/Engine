//
// Created by jpietrzyk on 15.07.19.
//

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_COMPARE_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_COMPARE_TSUBASA_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/compare.h>


namespace vectorlib {

   template< typename T >
   struct equal< aurora< v16k< T > >, 64 > {
      template<
         typename U = T,
         typename std::enable_if<
            std::is_integral< U >::value,
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< U > >::mask_t
      apply(
         typename aurora< v16k< U > >::vector_t const & p_Vec1,
         typename aurora< v16k< U > >::vector_t const & p_Vec2
      ) {
         _ve_lvl(256);
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: == ?. (aurora tsubasa)" );
         return _ve_vfmkl_mcv(VECC_EQ, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) );
      }
   };
   template< typename T >
   struct less< aurora< v16k< T > >, 64 > {
      template<
         typename U = T,
         typename std::enable_if<
            std::is_integral< U >::value,
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< U > >::mask_t
      apply(
         typename aurora< v16k< U > >::vector_t const & p_Vec1,
         typename aurora< v16k< U > >::vector_t const & p_Vec2
      ) {
         _ve_lvl(256);
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: < ?. (aurora tsubasa)" );
         return _ve_vfmkl_mcv(VECC_L, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) );
      }
   };
   template< typename T >
   struct lessequal< aurora< v16k< T > >, 64 > {
      template<
         typename U = T,
         typename std::enable_if<
            std::is_integral< U >::value,
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< U > >::mask_t
      apply(
         typename aurora< v16k< U > >::vector_t const & p_Vec1,
         typename aurora< v16k< U > >::vector_t const & p_Vec2
      ) {
         _ve_lvl(256);
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: <= ?. (aurora tsubasa)" );
         return _ve_vfmkl_mcv(VECC_LE, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) );
      }
   };
   template< typename T >
   struct greater< aurora< v16k< T > >, 64 > {
      template<
         typename U = T,
         typename std::enable_if<
            std::is_integral< U >::value,
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< U > >::mask_t
      apply(
         typename aurora< v16k< U > >::vector_t const & p_Vec1,
         typename aurora< v16k< U > >::vector_t const & p_Vec2
      ) {
         _ve_lvl(256);
         uint64_t * tmp = ( uint64_t * ) malloc( 256 * sizeof( uint64_t ) );
         uint64_t * tmp1 = ( uint64_t * ) malloc( 256 * sizeof( uint64_t ) );
         _ve_vst_vss( p_Vec1, sizeof( T ), tmp );
         _ve_vst_vss( p_Vec2, sizeof( T ), tmp1 );
         for( size_t j = 0; j < 256; j+=32 ) {
            for( size_t i = 0; i < 32; ++i ) {
               std::cout << (unsigned long)  tmp[ i + j ] << " | ";
            }
            std::cout << "\n";
            for( size_t i = 0; i < 32; ++i ) {
               std::cout << (unsigned long) tmp1[ i + j ] << " | ";
            }
            std::cout << "\n\n";
         }
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: > ?. (aurora tsubasa)" );
         std::cout << "Before Return: " << (unsigned long) count_matches<aurora<v16k<T>>>::apply( _ve_vfmkl_mcv(VECC_G, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) ) ) << "\n";
         std::cout << "Before Return: " << (unsigned long) _ve_pcvm_sm( _ve_vfmkl_mcv(VECC_G, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) ) ) << "\n";
         return _ve_vfmkl_mcv(VECC_G, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) );
      }
   };
   template< typename T >
   struct greaterequal< aurora< v16k< T > >, 64 > {
      template<
         typename U = T,
         typename std::enable_if<
            std::is_integral< U >::value,
            int
         >::type = 0
      >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename aurora< v16k< U > >::mask_t
      apply(
         typename aurora< v16k< U > >::vector_t const & p_Vec1,
         typename aurora< v16k< U > >::vector_t const & p_Vec2
      ) {
         _ve_lvl(256);
         trace( "[VECTOR] - Compare 64 bit integer values from two registers: >= ?. (aurora tsubasa)" );
         return _ve_vfmkl_mcv(VECC_GE, _ve_vcmpsl_vvv( p_Vec1, p_Vec2 ) );
      }
   };

   template< typename T >
   struct count_matches< aurora< v16k< T > > > {
   MSV_CXX_ATTRIBUTE_FORCE_INLINE
   static uint64_t
   apply(
      typename aurora< v16k< T > >::mask_t const p_Mask
   ) {
      _ve_lvl(256);
      std::cout << "IN COUNT MATCHES: Returning " << (unsigned long) _ve_pcvm_sm( p_Mask ) << "\n";
      trace( "[VECTOR] - Count matches in a comparison mask. (aurora tsubasa)" );
      return _ve_pcvm_sm( p_Mask );
   }
};
}

#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_COMPARE_TSUBASA_H
