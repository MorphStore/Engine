//
// Created by jpietrzyk on 15.04.19.
//

#ifndef MORPHSTORE_GROUP_UNCOMPR_H
#define MORPHSTORE_GROUP_UNCOMPR_H


#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>
#include <core/utils/preprocessor.h>
#include <core/operators/interfaces/group.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/processing_style.h>

#include <immintrin.h>
#include <core/utils/printing.h>

namespace morphstore {



template< int32_t N >
MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i rotl64( __m256i p_data, std::integral_constant< int32_t, N > ) {
   return _mm256_or_si256(
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
// for 8-byte values with seed
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p_values, uint64_t p_seed ) {
      return avalance(
         _mm256_add_epi64(
            mul64(
               rotl64(
                  _mm256_xor_si256(
                     _mm256_set1_epi64x( p_seed + PRIME64_5 + 8ULL ),
                     round64( p_values )
                  ),
                  IMM_INT32(27)
               ),
               IMM_UINT64(PRIME64_1)
            ),
            _mm256_set1_epi64x(PRIME64_4)
         )
      );
   }
// for 16-byte values with seed
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p1_values, __m256i p2_values, uint64_t p_seed ) {
      return avalance(
         _mm256_add_epi64(
            mul64(
               rotl64(
                  _mm256_xor_si256(
                     _mm256_add_epi64(
                        mul64(
                           rotl64(
                              _mm256_xor_si256(
                                 _mm256_set1_epi64x( p_seed + PRIME64_5 + 8ULL ),
                                 round64( p1_values )
                              ),
                              IMM_INT32(27)
                           ),
                           IMM_UINT64(PRIME64_1)
                        ),
                        _mm256_set1_epi64x(PRIME64_4)
                     ),
                     round64( p2_values )
                  ),
                  IMM_INT32(27)
               ),
               IMM_UINT64(PRIME64_1)
            ),
            _mm256_set1_epi64x(PRIME64_4)
         )
      );
   }
// for 8-byte values without seed
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p_values ) {
      __m256i v64 = _mm256_xor_si256(
         _mm256_set1_epi64x( PRIME64_5 + 8ULL ),
         round64( p_values ) );
      return avalance(_mm256_add_epi64( mul64( rotl64(v64, IMM_INT32(27)), IMM_UINT64(PRIME64_1)), _mm256_set1_epi64x(PRIME64_4) ));
   }
// for 16-byte values with seed
   static MSV_CXX_ATTRIBUTE_FORCE_INLINE __m256i apply( __m256i p1_values, __m256i p2_values ) {
      return avalance(
         _mm256_add_epi64(
            mul64(
               rotl64(
                  _mm256_xor_si256(
                     _mm256_add_epi64(
                        mul64(
                           rotl64(
                              _mm256_xor_si256(
                                 _mm256_set1_epi64x( PRIME64_5 + 8ULL ),
                                 round64( p1_values )
                              ),
                              IMM_INT32(27)
                           ),
                           IMM_UINT64(PRIME64_1)
                        ),
                        _mm256_set1_epi64x(PRIME64_4)
                     ),
                     round64( p2_values )
                  ),
                  IMM_INT32(27)
               ),
               IMM_UINT64(PRIME64_1)
            ),
            _mm256_set1_epi64x(PRIME64_4)
         )
      );
   }
};


template<>
const std::tuple<
   const column<uncompr_f> *,
   const column<uncompr_f> *
>
   group<processing_style_t::vec256>(
   column<uncompr_f> const * const  inDataCol,
   size_t            const          outExtCountEstimate
) {
   const size_t inDataCount = inDataCol->get_count_values();
   const size_t inDataSize = inDataCol->get_size_used_byte();

   const size_t outCount = bool(outExtCountEstimate) ? (outExtCountEstimate): inDataCount;
   auto outGrCol = new column<uncompr_f>(inDataSize);
   auto outExtCol = new column<uncompr_f>( outCount * sizeof( uint64_t ) );
   uint64_t * outGr = outGrCol->get_data();
   uint64_t * outExt = outExtCol->get_data();
   uint64_t * const initOutExt = outExt;

   const size_t hashContainerSize = 1.6 * outCount;
   const size_t lastPossibleVectorNumberInHashContainer = hashContainerSize - (sizeof( __m256i ) / sizeof( uint64_t ));
   uint64_t * hashContainerData = ( uint64_t * ) malloc( hashContainerSize * sizeof( uint64_t ) );
   uint64_t * hashContainerGroupIds = ( uint64_t * ) malloc( hashContainerSize * sizeof( uint64_t ) );

   uint64_t const * inData = inDataCol->get_data();

   uint64_t * tmpHashes = ( uint64_t * ) _mm_malloc( sizeof( __m256i ), sizeof( __m256i ) );
   __m256i const zeroV = _mm256_setzero_si256();

   const size_t elementCount = ( sizeof( __m256i ) / sizeof( uint64_t ) );

   size_t scalarPartCount = inDataCount & (elementCount-1);
   size_t vectorizedPartCount = inDataCount - scalarPartCount;
   uint64_t groupId = 0;

   for( size_t dataPos = 0; dataPos < vectorizedPartCount; dataPos += sizeof( __m256i ) / sizeof( uint64_t ) ) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(tmpHashes),
                         xxhash<uint64_t>::apply(_mm256_load_si256(reinterpret_cast<__m256i const *>(&(inData[dataPos])))));
      for(size_t hashPos = 0; hashPos < 4; ++hashPos) {
         uint32_t searchOffset = 0;
         bool dataFound;

         size_t pos = tmpHashes[hashPos] % lastPossibleVectorNumberInHashContainer;
         size_t pos_new = pos;
         __m256i dataV = _mm256_set1_epi64x(inData[dataPos+hashPos]);
         do {
            pos = pos_new;
            __m256i groupsV = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(&(hashContainerData[pos])));
            dataFound = false;

            pos_new = ( pos_new + sizeof(__m256i) / sizeof(uint64_t) ) % lastPossibleVectorNumberInHashContainer;
            searchOffset =
               _mm256_movemask_pd(
                  _mm256_castsi256_pd(
                     _mm256_cmpeq_epi64(
                        dataV,
                        groupsV
                     )
                  )
               );
            if(searchOffset == 0) {
               searchOffset =
                  _mm256_movemask_pd(
                     _mm256_castsi256_pd(
                        _mm256_cmpeq_epi64(
                           zeroV,
                           groupsV
                        )
                     )
                  );
            } else {
               dataFound = true;
            }
         } while(searchOffset == 0);
         pos += __builtin_ctz(searchOffset) % lastPossibleVectorNumberInHashContainer ;
         if(dataFound) {
            //_bit_scan_forward( searchResult ) SHOULD work... but it doesn't
            *(outGr++) = hashContainerGroupIds[pos]; //BS
         } else {
            hashContainerData[pos] = inData[dataPos+hashPos];
            hashContainerGroupIds[pos] = groupId;
            *(outGr++) = groupId++;
            *(outExt++) = dataPos + hashPos;
         }
      }
   }

   for( size_t dataPos = vectorizedPartCount; dataPos < inDataCount; ++dataPos ) {

   }

   const size_t outExtCount = outExt - initOutExt;
   outGrCol->set_meta_data(inDataCount, inDataSize);
   outExtCol->set_meta_data(outExtCount, outExtCount * sizeof(uint64_t));

   return std::make_tuple(outGrCol, outExtCol);
}



template<>
const std::tuple<
   const column<uncompr_f> *,
   const column<uncompr_f> *
>
group<processing_style_t::vec256>(
   const column<uncompr_f> * const inGrCol,
   const column<uncompr_f> * const inDataCol,
   const size_t outExtCountEstimate
) {
   const size_t inDataCount = inDataCol->get_count_values();
   const size_t inDataSize = inDataCol->get_size_used_byte();

   if(inDataCount != inGrCol->get_count_values())
      throw std::runtime_error(
         "binary group: inGrCol and inDataCol must contain the same "
         "number of data elements"
      );


   const size_t outCount = bool(outExtCountEstimate) ? (outExtCountEstimate): inDataCount;
   auto outGrCol = new column<uncompr_f>(inDataSize);
   auto outExtCol = new column<uncompr_f>( outCount * sizeof( uint64_t ) );
   uint64_t * outGr = outGrCol->get_data();
   uint64_t * outExt = outExtCol->get_data();
   uint64_t * const initOutExt = outExt;

   const size_t hashContainerSize = 1.6 * outCount;
   const size_t lastPossibleVectorNumberInHashContainer = hashContainerSize - (sizeof( __m256i ) / sizeof( uint64_t ));
   uint64_t * hashContainerData = ( uint64_t * ) malloc( hashContainerSize * sizeof( uint64_t ) );
   uint64_t * hashContainerInGr = ( uint64_t * ) malloc( hashContainerSize * sizeof( uint64_t ) );
   uint64_t * hashContainerGroupIds = ( uint64_t * ) malloc( hashContainerSize * sizeof( uint64_t ) );

   uint64_t const * const inData = inDataCol->get_data();
   uint64_t const * const inGr = inGrCol->get_data();

   uint64_t * tmpHashes = ( uint64_t * ) _mm_malloc( sizeof( __m256i ), sizeof( __m256i ) );
   __m256i const zeroV = _mm256_setzero_si256();

   const size_t elementCount = ( sizeof( __m256i ) / sizeof( uint64_t ) );

   size_t scalarPartCount = inDataCount & (elementCount-1);
   size_t vectorizedPartCount = inDataCount - scalarPartCount;
   uint64_t groupId = 0;

   for( size_t dataPos = 0; dataPos < vectorizedPartCount; dataPos += sizeof( __m256i ) / sizeof( uint64_t ) ) {
      _mm256_store_si256(reinterpret_cast<__m256i *>(tmpHashes),
                         xxhash<uint64_t>::apply(
                            _mm256_load_si256(reinterpret_cast<__m256i const *>(&(inData[dataPos]))),
                            _mm256_load_si256(reinterpret_cast<__m256i const *>(&(inGr[dataPos]))))
                         );
      for(size_t hashPos = 0; hashPos < 4; ++hashPos) {
         uint32_t searchOffset = 0;
         bool dataFound;
         size_t pos = tmpHashes[hashPos] % lastPossibleVectorNumberInHashContainer;
         size_t pos_new = pos;
         __m256i dataV = _mm256_set1_epi64x(inData[dataPos+hashPos]);
         __m256i groupIdV = _mm256_set1_epi64x(inGr[dataPos+hashPos]);
         do {
            pos = pos_new;
            __m256i hashedDataV = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(&(hashContainerData[pos])));
            __m256i hashGroupIdV = _mm256_loadu_si256(reinterpret_cast<__m256i const *>(&(hashContainerInGr[pos])));
            dataFound = false;

            pos_new = ( pos_new + sizeof(__m256i) / sizeof(uint64_t) ) % lastPossibleVectorNumberInHashContainer;
#if MSV_OPTIMIZE_GROUPBY_BINARY_TRANSFER_RESULT==1
            //immidiate Transfer
            searchOffset =
                  _mm256_movemask_pd(
                     _mm256_castsi256_pd(
                        _mm256_cmpeq_epi64(
                           dataV,
                           groupsV
                        )
                     )
                  )
               &
                  _mm256_movemask_pd(
                     _mm256_castsi256_pd(
                        _mm256_cmpeq_epi64(
                           groupV,
                           groupsIdV
                        )
                     )
                  );
#else
            searchOffset =
               _mm256_movemask_pd(
                  _mm256_castsi256_pd(
                     _mm256_and_si256(
                        _mm256_cmpeq_epi64(
                           dataV,
                           hashedDataV
                        ),
                        _mm256_cmpeq_epi64(
                           groupIdV,
                           hashGroupIdV
                        )
                     )
                  )
               );
#endif
            if(searchOffset == 0) {
#if MSV_OPTIMIZE_GROUPBY_BINARY_TRANSFER_RESULT==1
               searchOffset =
                     _mm256_movemask_pd(
                        _mm256_castsi256_pd(
                           _mm256_cmpeq_epi64(
                              zeroV,
                              groupsV
                           )
                        )
                     )
                  &
                     _mm256_movemask_pd(
                        _mm256_castsi256_pd(
                           _mm256_cmpeq_epi64(
                              zeroV,
                              groupsIdV
                           )
                        )
                     );
#else
               searchOffset =
                  _mm256_movemask_pd(
                     _mm256_castsi256_pd(
                        _mm256_and_si256(
                           _mm256_cmpeq_epi64(
                              zeroV,
                              hashedDataV
                           ),
                           _mm256_cmpeq_epi64(
                              zeroV,
                              hashGroupIdV
                           )
                        )
                     )
                  );
#endif
            } else {
               dataFound = true;
            }
         } while(searchOffset == 0);
         //_bit_scan_forward( searchResult ) SHOULD work... but it  doesn't
         pos += __builtin_ctz(searchOffset) % lastPossibleVectorNumberInHashContainer;
         if(dataFound) {
            *(outGr++) = hashContainerGroupIds[pos]; //BS
         } else {
            hashContainerData[pos] = inData[dataPos+hashPos];
            hashContainerInGr[pos] = inGr[dataPos+hashPos];
            hashContainerGroupIds[pos] = groupId;
            *(outGr++) = groupId++;
            *(outExt++) = dataPos + hashPos;
         }
      }
   }
   for( size_t dataPos = vectorizedPartCount; dataPos < inDataCount; ++dataPos ) {

   }

   const size_t outExtCount = outExt - initOutExt;
   outGrCol->set_meta_data(inDataCount, inDataSize);
   outExtCol->set_meta_data(outExtCount, outExtCount * sizeof(uint64_t));

   return std::make_tuple(outGrCol, outExtCol);
}


}






#endif //MORPHSTORE_GROUP_UNCOMPR_H
