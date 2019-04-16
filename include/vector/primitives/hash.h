//
// Created by jpietrzyk on 15.04.19.
//

#ifndef MORPHSTORE_HASH_H
#define MORPHSTORE_HASH_H

#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>
#include <core/utils/preprocessor.h>

#include <immintrin.h>

namespace morphstore {



template< int32_t N >
MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i rotl64( __m256i p_data, std::integral_constant< int32_t, N > ) {
   return _mm256_or_epi64(
         _mm256_slli_epi64( p_data, N ),
         _mm256_srli_epi64( p_data, 64-N)
      );
}

template< uint64_t N >
MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i mul64( __m256i p_a, std::integral_constant< uint64_t, N > ) {
   __m256i b = _mm256_set1_epi64x( N );
   __m256i c = _mm256_mullo_epi32( p_a, _mm256_shuffle_epi32( b, 0xB1 ) );
   return _mm256_add_epi64(
      _mm256_mul_epu32( p_a, b ),
      _mm256_and_si256(
         _mm256_add_epi32(
            _mm256_srli_epi64( c, 32 ),
            c
         ),
         _mm256_set1_epi64x( 0x00000000FFFFFFFF )
      )
   );
}
MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i mul64_single_b( __m256i p_a, __m256i p_b ) {
   __m256i c = _mm256_mullo_epi32( p_a, _mm256_shuffle_epi32( p_b, 0xB1 ) );
   return _mm256_add_epi64(
      _mm256_mul_epu32( p_a, p_b ),
      _mm256_and_si256(
         _mm256_add_epi32(
            _mm256_srli_epi64( c, 32 ),
            c
         ),
         _mm256_set1_epi64x( 0x00000000FFFFFFFF )
      )
   );
}
MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i mul64( __m256i p_a, __m256i p_b ) {
   __m256i bswap = _mm256_shuffle_epi32(p_b, 0xB1);           // swap H<->L
   __m256i prodlh = _mm256_mullo_epi32(p_a, bswap);            // 32 bit L*H products
   __m256i zero = _mm256_setzero_si256();                 // 0
   __m256i prodlh2 = _mm256_hadd_epi32(prodlh, zero);         // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
   __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2, 0x73);     // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
   __m256i prodll = _mm256_mul_epu32(p_a, p_b);                  // a0Lb0L,a1Lb1L, 64 bit unsigned products
   __m256i prod = _mm256_add_epi64(prodll, prodlh3);       // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
   return prod;
}

template< typename T >
struct xxhash {
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p_values, T p_see ) = delete;
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p_values ) = delete;
};

template< >
struct xxhash< uint64_t > {
   static const uint64_t PRIME64_1 = 11400714785074694791ULL;   /* 0b1001111000110111011110011011000110000101111010111100101010000111 */
   static const uint64_t PRIME64_2 = 14029467366897019727ULL;   /* 0b1100001010110010101011100011110100100111110101001110101101001111 */
   static const uint64_t PRIME64_3 =  1609587929392839161ULL;   /* 0b0001011001010110011001111011000110011110001101110111100111111001 */
   static const uint64_t PRIME64_4 =  9650029242287828579ULL;   /* 0b1000010111101011110010100111011111000010101100101010111001100011 */
   static const uint64_t PRIME64_5 =  2870177450012600261ULL;   /* 0b0010011111010100111010110010111100010110010101100110011111000101 */

   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i round64( __m256i p_acc, __m256i p_value ) {
      return
         mul64(
            rotl64(
               _mm256_add_epi64(
                  p_acc,
                  mul64( p_value, IMM_UINT64( PRIME64_2 ) )
               ),
               IMM_INT32( 31 )
            ),
            IMM_UINT64(PRIME64_1)
         );
   }
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i round64( __m256i p_value ) {
      return
         mul64(
            rotl64(
               mul64( p_value, IMM_UINT64( PRIME64_2 ) ),
               IMM_INT32( 31 )
            ),
            IMM_UINT64(PRIME64_1)
         );
   }

   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i merge_round_64( __m256i p_acc, __m256i p_value ) {
      return
         _mm256_add_epi64(
            mul64(
               _mm256_xor_si256(
                  p_acc,
                  round64( p_value )
               ),
               IMM_UINT64(PRIME64_1)
            ),
            _mm256_set1_epi64x( PRIME64_4 )
         );
   }

   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i avalance( __m256i p_value ) {
      __m256i tmp = mul64( _mm256_xor_si256( p_value, _mm256_srli_epi64( p_value, 33 ) ), IMM_UINT64( PRIME64_2 ) );
      __m256i tmp1 = mul64( _mm256_xor_si256( tmp, _mm256_srli_epi64( tmp, 29 ) ), IMM_UINT64( PRIME64_3 ) );
      return ( _mm256_xor_si256( tmp1, _mm256_srli_epi64( tmp1, 32 ) ) );

   }

// see https://github.com/Cyan4973/xxHash
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p_values, uint64_t p_seed ) {
      __m256i v64 = _mm256_xor_si256(
         _mm256_set1_epi64x( p_seed + PRIME64_5 + 8ULL ),
         round64( p_values ) );
      return avalance(_mm256_add_epi64( mul64( rotl64(v64, IMM_INT32(27)), IMM_UINT64(PRIME64_1)), _mm256_set1_epi64x(PRIME64_4) ));
   }

// see https://github.com/aappleby/smhasher/wiki/MurmurHash3
// http://bitsquid.blogspot.com/2011/08/code-snippet-murmur-hash-inverse-pre.html
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p_values ) {
      __m256i v64 = _mm256_xor_si256(
         _mm256_set1_epi64x( PRIME64_5 + 8ULL ),
         round64( p_values ) );
      return avalance(_mm256_add_epi64( mul64( rotl64(v64, IMM_INT32(27)), IMM_UINT64(PRIME64_1)), _mm256_set1_epi64x(PRIME64_4) ));
   }
};


MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256 murmur

#endif //MORPHSTORE_HASH_H