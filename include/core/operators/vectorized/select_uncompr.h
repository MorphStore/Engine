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
 * 1. This does not work for very large unsigned integers, which use all 64 bits, because values are casted to signed.
 * 2. 256-bit-version does not (yet) store indices correctly when there are gaps within one register
 */

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

        
        
        __m128i value = _mm_set_epi64x(val,val);
        __m128i cmpres;
        int mask;
        
        if (typeid(op)==typeid(std::less<uint64_t>)){
         
            for(unsigned i = 0; i < inDataCount/2; i++){
              
                cmpres = _mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] ));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));
                
            }
        }
        
        if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            for(unsigned i = 0; i < inDataCount/2; i++){

                cmpres = _mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value);
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));

            }
        }
        
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            
            for(unsigned i = 0; i < inDataCount/2; i++){
                
                cmpres = _mm_cmpeq_epi64(_mm_load_si128( &inData[i]) ,value);
                mask = _mm_movemask_pd((__m128d)cmpres);
                if ((mask << 31)==0) outPos--;
                // Why does this the normal pointer magic not work here for equality???
                _mm_maskstore_epi64(outPos, cmpres,_mm_set_epi64x(i*2+1,i*2));
                if ((mask << 31)==0) outPos++;
                outPos+=__builtin_popcountl(mask);
            }
        }
        
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
         
            for(unsigned i = 0; i < inDataCount/2; i++){
              
                __m128i cmpres = _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
        
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){

            for(unsigned i = 0; i < inDataCount/2; i++){
          
                __m128i cmpres = _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] )));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos=(long long int *)((uint64_t *)outPos-(__builtin_clz(mask)-30));
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos=(long long int *)((uint64_t *)outPos+(__builtin_clz(mask)-30));
                outPos=(long long int *)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
      
        const size_t outPosCount = (uint64_t*)outPos - (uint64_t*)initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
       return outPosCol; 
    }
};

/* This does not really compress, just shift the values we want to store to the lower bits.
 * If you need a real compress store, copy this code and change the used store-intrinsic to _mm256_maskstore* (and provide the according mask, of course).
 * This function will move to the vector lib someday.
 */
MSV_CXX_ATTRIBUTE_FORCE_INLINE/*__attribute__((always_inline)) inline*/ void compress_store256(__m256i * outPtr, int mask, __m256i vector){
    switch (mask){
          
                    case 0: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 1: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 2: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,225)); break;
                    case 3: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,228)); break;
                    case 4: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,210)); break;
                    case 5: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,216)); break;
                    case 6: _mm256_storeu_si256(outPtr, _mm256_permute4x64_epi64(vector,201)); break;
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

        
        
        __m256i value = _mm256_set_epi64x(val,val,val,val);
        __m256i cmpres;
        __m256i ids=_mm256_set_epi64x(3,2,1,0);
        __m256i add=_mm256_set_epi64x(4,4,4,4);
        int mask;
        
        if (typeid(op)==typeid(std::less<uint64_t>)){
         
            for(unsigned i = 0; i < inDataCount/4; i++){
              
                cmpres = _mm256_cmpgt_epi64(value,_mm256_load_si256( &inData[i] ));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                
                compress_store256(outPos,mask,ids);
                
                ids=_mm256_add_epi64(ids,add);
                
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
                
            }
        }
        
        if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            for(unsigned i = 0; i < inDataCount/4; i++){

                cmpres = _mm256_cmpgt_epi64(_mm256_load_si256( &inData[i] ),value);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);      
                
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));

            }
        }
        
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            
            for(unsigned i = 0; i < inDataCount/4; i++){
                
                cmpres = _mm256_cmpeq_epi64(_mm256_load_si256( &inData[i]) ,value);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
                
            }
        }
        
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
         
            for(unsigned i = 0; i < inDataCount/4; i++){
              
                cmpres = _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_load_si256( &inData[i] ),value),_mm256_cmpgt_epi64(_mm256_load_si256( &inData[i] ),value));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
        
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){

            for(unsigned i = 0; i < inDataCount/4; i++){
          
                cmpres = _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_load_si256( &inData[i] ),value),_mm256_cmpgt_epi64(value,_mm256_load_si256( &inData[i] )));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                compress_store256(outPos,mask,ids);
                ids=_mm256_add_epi64(ids,add);
                outPos=(__m256i*)((uint64_t *)outPos+__builtin_popcountl(mask));
            }
        }
      
        const size_t outPosCount = ((uint64_t *)outPos - (uint64_t *)initOutPos);
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
       return outPosCol; 
    }
};


    }

#endif /* SELECT_UNCOMPR_H */

