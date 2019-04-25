/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   select_uncompr.h
 * Author: Annett
 *
 * Created on 21. März 2019, 12:21
 */

#ifndef SELECT_UNCOMPR_H
#define SELECT_UNCOMPR_H

#include <core/operators/interfaces/select.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

#include <cstdint>
#include <immintrin.h>
#include <typeinfo>

namespace morphstore {
    
/*!
 * Known issues
 * - This Select does not work for very large unsigned integers, which use all 64 bits, because values are casted to signed.
 * 
 */

    //! 128-bit Select implementation using only SSE and AVX(2) intrinsics
template<template<typename> class t_op>
struct select<t_op, processing_style_t::vec128, uncompr_f, uncompr_f> {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        const size_t inDataCount = inDataCol->get_count_values();
        const __m128i * const inData = inDataCol->get_data();
        auto outPosCol = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : inDataCol->get_size_used_byte()
        );
        
        t_op<uint64_t> op;
        uint64_t * outP =  outPosCol->get_data();
        //I know the following is ugly, but _mm_maskstore_epi64 requires a long long (64 bit types are only long on a 64-bit system))
        long long int * outPos =  reinterpret_cast< long long int * >(outP);
        const long long int * const initOutPos = reinterpret_cast< long long int * > (outP);

        
        __m128i value = _mm_set_epi64x(val,val);//! Fill a register with the select predicate
        __m128i cmpres;
        int mask;
        unsigned i=0;
        
        //!Are we doing a less-than comparison?
        if (typeid(op)==typeid(std::less<uint64_t>)){
         
            //!Iterate over all elements of the column, 2 in each iteration because we can store 2 64-bit values in a 128 bit register
            for( i = 0; i < inDataCount/2; i++){
              
                /*!
                 * Do the following steps for comparison:
                 * 1. Load 2 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Pointer magic: We move the output position one element in front of the actual output position if the first of our values (loaded in step 1) did not match. We nee dthis step to not produce "holes" in step 4.
                 * 4. maskstore: Store the index for each value that matched in step 1 
                 * 5. Move the output position bakc if we changed it in step 3
                 * 6. Increase output position again by the number of matching results in this step
                 */
                cmpres = _mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] ));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));
                
            }
        }
        
        //Are we doing a greater-than comparison?
        if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            for( i = 0; i < inDataCount/2; i++){

                                /*!
                 * Do the following steps for comparison:
                 * 1. Load 2 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Pointer magic: We move the output position one element in front of the actual output position if the first of our values (loaded in step 1) did not match. We nee dthis step to not produce "holes" in step 4.
                 * 4. maskstore: Store the index for each value that matched in step 1 
                 * 5. Move the output position bakc if we changed it in step 3
                 * 6. Increase output position again by the number of matching results in this step
                 */
                cmpres = _mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value);
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));

            }
        }
        
        //Are we comparing for equality?
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            
            for( i = 0; i < inDataCount/2; i++){
                
                                /*!
                 * Do the following steps for comparison:
                 * 1. Load 2 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Pointer magic: We move the output position one element in front of the actual output position if the first of our values (loaded in step 1) did not match. We nee dthis step to not produce "holes" in step 4.
                 * 4. maskstore: Store the index for each value that matched in step 1 
                 * 5. Move the output position bakc if we changed it in step 3
                 * 6. Increase output position again by the number of matching results in this step
                 */
                cmpres = _mm_cmpeq_epi64(_mm_load_si128( &inData[i]) ,value);
                mask = _mm_movemask_pd((__m128d)cmpres);
                if ((mask << 31)==0) outPos--;
                // Why does this the normal pointer magic not work here for equality???
                _mm_maskstore_epi64(outPos, cmpres,_mm_set_epi64x(i*2+1,i*2));
                if ((mask << 31)==0) outPos++;
                outPos+=__builtin_popcountl(mask);
            }
        }
        
        
        //Are we doing a greater-than-or-equal comparison?
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
         
            for( i = 0; i < inDataCount/2; i++){
              
                                /*!
                 * Do the following steps for comparison:
                 * 1. Load 2 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Pointer magic: We move the output position one element in front of the actual output position if the first of our values (loaded in step 1) did not match. We nee dthis step to not produce "holes" in step 4.
                 * 4. maskstore: Store the index for each value that matched in step 1 
                 * 5. Move the output position bakc if we changed it in step 3
                 * 6. Increase output position again by the number of matching results in this step
                 */
                __m128i cmpres = _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
        
        //!Are we doing a less-than-or-equal comparison?
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){

             //!Iterate over all elements of the column, 2 in each iteration because we can store 2 64-bit values in a 128 bit register
            for( i = 0; i < inDataCount/2; i++){
          
                                /*!
                 * Do the following steps for comparison:
                 * 1. Load 2 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Pointer magic: We move the output position one element in front of the actual output position if the first of our values (loaded in step 1) did not match. We nee dthis step to not produce "holes" in step 4.
                 * 4. maskstore: Store the index for each value that matched in step 1 
                 * 5. Move the output position bakc if we changed it in step 3
                 * 6. Increase output position again by the number of matching results in this step
                 */
                __m128i cmpres = _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] )));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
      
         uint64_t* oPos=(uint64_t*)outPos;
         for(unsigned j = i; j < inDataCount; j++)
            if(op(((uint64_t*)inData)[j], val)) {
                *oPos = j;
                oPos++;
            }
          
        const size_t outPosCount = (uint64_t*)oPos - (uint64_t*)initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
       return outPosCol; 
    }
};

/*! This function compresses the data in a register according to a bitmask (all values with an according set bit will be packed at the beginning of the register), and stores it at a given address.
 * Note: This does not really compress, just shift the values we want to store to the lower bits.
 * If you need a real compress store, copy this code and change the used store-intrinsic to _mm256_maskstore* (and provide the according mask, of course).
 * This function will move to the vector lib someday.
 * @param outPtr The memory address where the vector should be stored
 * @param mask A bitmask with a bit set for every value which is goin to be stored
 * @param vector The 256-bit vector to be comprssed and stored 
 */
MSV_CXX_ATTRIBUTE_FORCE_INLINE/*__attribute__((always_inline)) inline*/ void compress_store256(__m256i * outPtr, int mask, __m256i vector){
    switch (mask){
          
                    case 0: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 1: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 2: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,57)); break;
                    case 3: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 4: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,78)); break;
                    case 5: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,216)); break;
                    case 6: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,57)); break;
                    case 7: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 8: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,147)); break;
                    case 9: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,156)); break;
                    case 10: _mm256_storeu_si256(outPtr,_mm256_permute4x64_epi64(vector,141)); break;
                    case 11: _mm256_storeu_si256(outPtr,_mm256_permute4x64_epi64(vector,180)); break;
                    case 12: _mm256_storeu_si256(outPtr,_mm256_permute4x64_epi64(vector,78)); break;
                    case 13: _mm256_storeu_si256(outPtr,_mm256_permute4x64_epi64(vector,120)); break;
                    case 14: _mm256_storeu_si256(outPtr,_mm256_permute4x64_epi64(vector,57)); break;
                    case 15: _mm256_storeu_si256(outPtr,_mm256_permute4x64_epi64(vector,228)); break;
                }
}

//! 256-bit Select implementation using only SSE and AVX(2) intrinsics
template<template<typename> class t_op>
struct select<t_op, processing_style_t::vec256, uncompr_f, uncompr_f> {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        const size_t inDataCount = inDataCol->get_count_values();
        const __m256i * const inData = inDataCol->get_data();
        auto outPosCol = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : inDataCol->get_size_used_byte()
        );
        
        t_op<uint64_t> op;
        uint64_t * outP =  outPosCol->get_data();
        //I know the following is ugly, but _mm_maskstore_epi64 requires a long long (64 bit types are only long on a 64-bit system))
        __m256i * outPos =  reinterpret_cast< __m256i * >(outP);
        const __m256i * const initOutPos = reinterpret_cast< __m256i * > (outP);

        
        
        __m256i value = _mm256_set_epi64x(val,val,val,val);//!Fill a vector with the select predicate
        __m256i cmpres;
        __m256i ids=_mm256_set_epi64x(3,2,1,0);//!Set initial IDs
        __m256i add=_mm256_set_epi64x(4,4,4,4);//!We will use this vector later to increase the IDs in every iteration
        int mask;
        unsigned i=0;
        //!Are we doing a less-than comparison?
        if (typeid(op)==typeid(std::less<uint64_t>)){
         
            for( i = 0; i < inDataCount/4; i++){
              
                 /*!
                 * Do the following steps for comparison:
                 * 1. Load 4 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Store the index for each value that matched in step 1 
                 * 4. Increase all IDs by 4 (maybe we should do it this way for 128 bit, too)
                 * 5. Increase output position by the number of results in this step
                 */
                cmpres = _mm256_cmpgt_epi64(value,_mm256_load_si256( &inData[i] ));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                
                compress_store256(outPos,mask,ids);
                
                ids=_mm256_add_epi64(ids,add);
                
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
                
            }
        }
        
        //!Are we doing a greater-than comparison?
        if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            //!Iterate over all elements of the column, 4 in each iteration because we can store 4 64-bit values in a 128 bit register
            for( i = 0; i < inDataCount/4; i++){

                   /*!
                 * Do the following steps for comparison:
                 * 1. Load 4 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Store the index for each value that matched in step 1 
                 * 4. Increase all IDs by 4 (maybe we should do it this way for 128 bit, too)
                 * 5. Increase output position by the number of results in this step
                 */
                cmpres = _mm256_cmpgt_epi64(_mm256_load_si256( &inData[i] ),value);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);      
                
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));

            }
        }
        
        //Are we comparing for equality?
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            //!Iterate over all elements of the column, 4 in each iteration because we can store 4 64-bit values in a 128 bit register
            for( i = 0; i < inDataCount/4; i++){
                
                /*!
                 * Do the following steps for comparison:
                 * 1. Load 4 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Store the index for each value that matched in step 1 
                 * 4. Increase all IDs by 4 (maybe we should do it this way for 128 bit, too)
                 * 5. Increase output position by the number of results in this step
                 */
                cmpres = _mm256_cmpeq_epi64(_mm256_load_si256( &inData[i]) ,value);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
                
            }
        }
        
        //Are we doing a greater-than-or-equal comparison?
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
         //!Iterate over all elements of the column, 4 in each iteration because we can store 4 64-bit values in a 128 bit register
            for( i = 0; i < inDataCount/4; i++){
              
                /*!
                 * Do the following steps for comparison:
                 * 1. Load 4 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Store the index for each value that matched in step 1 
                 * 4. Increase all IDs by 4 (maybe we should do it this way for 128 bit, too)
                 * 5. Increase output position by the number of results in this step
                 */
                cmpres = _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_load_si256( &inData[i] ),value),_mm256_cmpgt_epi64(_mm256_load_si256( &inData[i] ),value));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
        
        //Are we doing a less-than-or-equal comparison?
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){

             //!Iterate over all elements of the column, 4 in each iteration because we can store 4 64-bit values in a 128 bit register
            for(i = 0; i < inDataCount/4; i++){
          
                /*!
                 * Do the following steps for comparison:
                 * 1. Load 4 values into vector register and compare them with the predicate 
                 * 2. Make a mask out of the result of step 1
                 * 3. Store the index for each value that matched in step 1 
                 * 4. Increase all IDs by 4 (maybe we should do it this way for 128 bit, too)
                 * 5. Increase output position by the number of results in this step
                 */
                cmpres = _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_load_si256( &inData[i] ),value),_mm256_cmpgt_epi64(value,_mm256_load_si256( &inData[i] )));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
      
        uint64_t* oPos=(uint64_t*)outPos;
         for(unsigned j = i; j < inDataCount; j++)
            if(op(((uint64_t*)inData)[j], val)) {
                oPos[0] = j;
                oPos++;
            }
        
        
        const size_t outPosCount = ((uint64_t *)oPos - (uint64_t *)initOutPos); //!<How large is our result set?
        
        //Store output size in meta data of the output column
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
       return outPosCol; 
    }
};


    }

#endif /* SELECT_UNCOMPR_H */

