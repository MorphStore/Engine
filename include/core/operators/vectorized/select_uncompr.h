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
                outPos-=(__builtin_clz(mask)-30);
                _mm_maskstore_epi64(outPos, cmpres, _mm_set_epi64x(i*2+1,i*2));
                outPos+=(__builtin_clz(mask)-30);
                outPos+=__builtin_popcountl(_mm_movemask_pd((__m128d)cmpres));
                
            }
        }
        
        if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            for(unsigned i = 0; i < inDataCount/2; i++){

                cmpres = _mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value);
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos-=(__builtin_clz(mask)-30);
                _mm_maskstore_epi64(outPos, cmpres, _mm_set_epi64x(i*2+1,i*2));
                outPos+=(__builtin_clz(mask)-30);
                outPos+=__builtin_popcountl(mask);

            }
        }
        
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            
            for(unsigned i = 0; i < inDataCount/2; i++){
                
                cmpres = _mm_cmpeq_epi64(_mm_load_si128( &inData[i]) ,value);
                mask = _mm_movemask_pd((__m128d)cmpres);
                if ((mask << 31)==0) outPos--;
                //outPos-=(__builtin_clz(mask)-30); <- why does this not work for equality???
                _mm_maskstore_epi64(outPos, cmpres,_mm_set_epi64x(i*2+1,i*2));
                //outPos+=(__builtin_clz(mask)-30);
                if ((mask << 31)==0) outPos++;
                outPos+=__builtin_popcountl(mask);
                
            }
        }
        
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
         
            for(unsigned i = 0; i < inDataCount/2; i++){
              
                __m128i cmpres = _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos-=(__builtin_clz(mask)-30);
                _mm_maskstore_epi64(outPos,cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos+=(__builtin_clz(mask)-30);
                outPos+=__builtin_popcountl(mask);
            }
        }
        
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){

            for(unsigned i = 0; i < inDataCount/2; i++){
          
                __m128i cmpres = _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] )));
                mask = _mm_movemask_pd((__m128d)cmpres);
                outPos-=(__builtin_clz(mask)-30);
                _mm_maskstore_epi64(outPos, cmpres , _mm_set_epi64x(i*2+1,i*2));
                outPos+=(__builtin_clz(mask)-30);
                outPos+=__builtin_popcountl(mask);
            }
        }
      
        const size_t outPosCount = outPos - initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
       return outPosCol; 
    }
};

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
        long long int * outPos =  reinterpret_cast< long long int * >(outP);
        const long long int * const initOutPos = reinterpret_cast< long long int * > (outP);

        
        
        __m256i value = _mm256_set_epi64x(val,val,val,val);
        __m256i cmpres;
        int mask;
        
        if (typeid(op)==typeid(std::less<uint64_t>)){
         
            for(unsigned i = 0; i < inDataCount/4; i++){
              
                cmpres = _mm256_cmpgt_epi64(value,_mm256_load_si256( &inData[i] ));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                outPos-=(__builtin_clz(mask)-28);
                _mm256_maskstore_epi64(outPos, cmpres, _mm256_set_epi64x(i*4+3,i*4+2,i*4+1,i*4));
                outPos+=(__builtin_clz(mask)-28);
                outPos+=__builtin_popcountl(mask);
                
            }
        }
        
        if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            for(unsigned i = 0; i < inDataCount/4; i++){

                cmpres = _mm256_cmpgt_epi64(_mm256_load_si256( &inData[i] ),value);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                outPos-=(__builtin_clz(mask)-28);
                _mm256_maskstore_epi64(outPos, cmpres, _mm256_set_epi64x(i*4+3,i*4+2,i*4+1,i*4));
                outPos+=(__builtin_clz(mask)-28);
                outPos+=__builtin_popcountl(mask);

            }
        }
        
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            
            for(unsigned i = 0; i < inDataCount/4; i++){
                
                cmpres = _mm256_cmpeq_epi64(_mm256_load_si256( &inData[i]) ,value);
                mask = _mm256_movemask_pd((__m256d)cmpres);
                outPos-=(__builtin_clz(mask)-28);
                _mm256_maskstore_epi64(outPos, cmpres, _mm256_set_epi64x(i*4+3,i*4+2,i*4+1,i*4));
                outPos+=(__builtin_clz(mask)-28);
                outPos+=__builtin_popcountl(mask);
                
            }
        }
        
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
         
            for(unsigned i = 0; i < inDataCount/4; i++){
              
                cmpres = _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_load_si256( &inData[i] ),value),_mm256_cmpgt_epi64(_mm256_load_si256( &inData[i] ),value));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                outPos-=(__builtin_clz(mask)-28);
                _mm256_maskstore_epi64(outPos, cmpres, _mm256_set_epi64x(i*4+3,i*4+2,i*4+1,i*4));
                outPos+=(__builtin_clz(mask)-28);
                outPos+=__builtin_popcountl(mask);
            }
        }
        
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){

            for(unsigned i = 0; i < inDataCount/4; i++){
          
                cmpres = _mm256_or_si256(_mm256_cmpeq_epi64(_mm256_load_si256( &inData[i] ),value),_mm256_cmpgt_epi64(value,_mm256_load_si256( &inData[i] )));
                mask = _mm256_movemask_pd((__m256d)cmpres);
                outPos-=(__builtin_clz(mask)-28);
                _mm256_maskstore_epi64(outPos, cmpres, _mm256_set_epi64x(i*4+3,i*4+2,i*4+1,i*4));
                outPos+=(__builtin_clz(mask)-28);
                outPos+=__builtin_popcountl(mask);
            }
        }
      
        const size_t outPosCount = outPos - initOutPos;
        outPosCol->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
       return outPosCol; 
    }
};
/*template<template<typename> class t_op>
struct select<t_op, processing_style_t::vec256, uncompr_f, uncompr_f> {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataCol,
            const uint64_t val,
            const size_t outPosCountEstimate = 0
    ) {
        
         auto outPosCol = new column<uncompr_f>(
                bool(outPosCountEstimate)
                // use given estimate
                ? (outPosCountEstimate * sizeof(uint64_t))
                // use pessimistic estimate
                : inDataCol->get_size_used_byte()
        );
         
         return outPosCol;
    }
};*/

    }

#endif /* SELECT_UNCOMPR_H */

