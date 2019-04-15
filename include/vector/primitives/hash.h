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

const uint64_t PRIME64_1 = 11400714785074694791ULL;   /* 0b1001111000110111011110011011000110000101111010111100101010000111 */
const uint64_t PRIME64_2 = 14029467366897019727ULL;   /* 0b1100001010110010101011100011110100100111110101001110101101001111 */
const uint64_t PRIME64_3 =  1609587929392839161ULL;   /* 0b0001011001010110011001111011000110011110001101110111100111111001 */
const uint64_t PRIME64_4 =  9650029242287828579ULL;   /* 0b1000010111101011110010100111011111000010101100101010111001100011 */
const uint64_t PRIME64_5 =  2870177450012600261ULL;   /* 0b0010011111010100111010110010111100010110010101100110011111000101 */

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

MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i round64( __m256i p_acc, __m256i p_value ) {
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
MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i round64( __m256i p_value ) {
   return
      mul64(
         rotl64(
            mul64( p_value, IMM_UINT64( PRIME64_2 ) ),
            IMM_INT32( 31 )
         ),
         IMM_UINT64(PRIME64_1)
      );
}

MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i merge_round_64( __m256i p_acc, __m256i p_value ) {
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

MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i avalance( __m256i p_value ) {
   __m256i tmp = mul64( _mm256_xor_si256( p_value, _mm256_srli_epi64( p_value, 33 ) ), IMM_UINT64( PRIME64_2 ) );
   __m256i tmp1 = mul64( _mm256_xor_si256( tmp, _mm256_srli_epi64( tmp, 29 ) ), IMM_UINT64( PRIME64_3 ) );
   return ( _mm256_xor_si256( tmp1, _mm256_srli_epi64( tmp1, 32 ) ) );

}

MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i xxhash64( __m256i p_values ) {

}

#define XXH_get64bits(p) XXH_readLE64_align(p, align)

   static uint64_t
   XXH64_finalize(uint64_t h64, const void* ptr, size_t len, XXH_alignment align)
   {
      const BYTE* p = (const BYTE*)ptr;

#define PROCESS1_64            \
    h64 ^= (*p++) * PRIME64_5; \
    h64 = XXH_rotl64(h64, 11) * PRIME64_1;

#define PROCESS4_64          \
    h64 ^= (uint64_t)(XXH_get32bits(p)) * PRIME64_1; \
    p+=4;                    \
    h64 = XXH_rotl64(h64, 23) * PRIME64_2 + PRIME64_3;

#define PROCESS8_64 {        \
    uint64_t const k1 = XXH64_round(0, XXH_get64bits(p)); \
    p+=8;                    \
    h64 ^= k1;               \
    h64  = XXH_rotl64(h64,27) * PRIME64_1 + PRIME64_4; \
   }

      switch(len&31) {
         case 24: PROCESS8_64;
            /* fallthrough */
         case 16: PROCESS8_64;
            /* fallthrough */
         case  8: PROCESS8_64;
            return XXH64_avalanche(h64);

         case 28: PROCESS8_64;
            /* fallthrough */
         case 20: PROCESS8_64;
            /* fallthrough */
         case 12: PROCESS8_64;
            /* fallthrough */
         case  4: PROCESS4_64;
            return XXH64_avalanche(h64);

         case 25: PROCESS8_64;
            /* fallthrough */
         case 17: PROCESS8_64;
            /* fallthrough */
         case  9: PROCESS8_64;
            PROCESS1_64;
            return XXH64_avalanche(h64);

         case 29: PROCESS8_64;
            /* fallthrough */
         case 21: PROCESS8_64;
            /* fallthrough */
         case 13: PROCESS8_64;
            /* fallthrough */
         case  5: PROCESS4_64;
            PROCESS1_64;
            return XXH64_avalanche(h64);

         case 26: PROCESS8_64;
            /* fallthrough */
         case 18: PROCESS8_64;
            /* fallthrough */
         case 10: PROCESS8_64;
            PROCESS1_64;
            PROCESS1_64;
            return XXH64_avalanche(h64);

         case 30: PROCESS8_64;
            /* fallthrough */
         case 22: PROCESS8_64;
            /* fallthrough */
         case 14: PROCESS8_64;
            /* fallthrough */
         case  6: PROCESS4_64;
            PROCESS1_64;
            PROCESS1_64;
            return XXH64_avalanche(h64);

         case 27: PROCESS8_64;
            /* fallthrough */
         case 19: PROCESS8_64;
            /* fallthrough */
         case 11: PROCESS8_64;
            PROCESS1_64;
            PROCESS1_64;
            PROCESS1_64;
            return XXH64_avalanche(h64);

         case 31: PROCESS8_64;
            /* fallthrough */
         case 23: PROCESS8_64;
            /* fallthrough */
         case 15: PROCESS8_64;
            /* fallthrough */
         case  7: PROCESS4_64;
            /* fallthrough */
         case  3: PROCESS1_64;
            /* fallthrough */
         case  2: PROCESS1_64;
            /* fallthrough */
         case  1: PROCESS1_64;
            /* fallthrough */
         case  0: return XXH64_avalanche(h64);
      }

      /* impossible to reach */
      assert(0);
      return 0;  /* unreachable, but some compilers complain without it */
   }

   XXH_FORCE_INLINE uint64_t
   XXH64_endian_align(const void* input, size_t len, uint64_t seed, XXH_alignment align)
   {
      const BYTE* p = (const BYTE*)input;
      const BYTE* bEnd = p + len;
      uint64_t h64;

#if defined(XXH_ACCEPT_NULL_INPUT_POINTER) && (XXH_ACCEPT_NULL_INPUT_POINTER>=1)
      if (p==NULL) {
        len=0;
        bEnd=p=(const BYTE*)(size_t)32;
    }
#endif

      if (len>=32) {
         const BYTE* const limit = bEnd - 32;
         uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
         uint64_t v2 = seed + PRIME64_2;
         uint64_t v3 = seed + 0;
         uint64_t v4 = seed - PRIME64_1;

         do {
            v1 = XXH64_round(v1, XXH_get64bits(p)); p+=8;
            v2 = XXH64_round(v2, XXH_get64bits(p)); p+=8;
            v3 = XXH64_round(v3, XXH_get64bits(p)); p+=8;
            v4 = XXH64_round(v4, XXH_get64bits(p)); p+=8;
         } while (p<=limit);

         h64 = XXH_rotl64(v1, 1) + XXH_rotl64(v2, 7) + XXH_rotl64(v3, 12) + XXH_rotl64(v4, 18);
         h64 = XXH64_mergeRound(h64, v1);
         h64 = XXH64_mergeRound(h64, v2);
         h64 = XXH64_mergeRound(h64, v3);
         h64 = XXH64_mergeRound(h64, v4);

      } else {
         h64  = seed + PRIME64_5;
      }

      h64 += (uint64_t) len;

      return XXH64_finalize(h64, p, len, align);
   }


   XXH_PUBLIC_API unsigned long long XXH64 (const void* input, size_t len, unsigned long long seed)
   {
#if 0
      /* Simple version, good for code maintenance, but unfortunately slow for small inputs */
    XXH64_state_t state;
    XXH64_reset(&state, seed);
    XXH64_update(&state, input, len);
    return XXH64_digest(&state);

#else

      if (XXH_FORCE_ALIGN_CHECK) {
         if ((((size_t)input) & 7)==0) {  /* Input is aligned, let's leverage the speed advantage */
            return XXH64_endian_align(input, len, seed, XXH_aligned);
         }   }

      return XXH64_endian_align(input, len, seed, XXH_unaligned);

#endif
   }

}


#endif //MORPHSTORE_HASH_H
