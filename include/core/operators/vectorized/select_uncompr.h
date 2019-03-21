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
 * Known issue: This does not work for very large usigned integers, which use all 64 bits, because values are casted to signed.
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
        
        if (typeid(op)==typeid(std::less<uint64_t>)){
            
            _mm_maskstore_epi64(outPos, _mm_cmpgt_epi64(value,_mm_load_si128( &inData[0] )), _mm_set_epi64x(0,1));
                
                if (__builtin_popcountl(*outPos)) outPos++;
                if (__builtin_popcountl(*outPos)) outPos++;
            
            for(unsigned i = 1; i < inDataCount/2; i++){
                //cmpgt -> compare
                //maskstore vector of i (mask is result of cmpgt)
                _mm_maskstore_epi64(outPos, _mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] )), _mm_set_epi64x(i/2,i/2+1));
                
                if (*outPos) outPos++;
                if (*outPos) outPos++;
            }
        }
        
           if (typeid(op)==typeid(std::greater<uint64_t>)){
            
            _mm_maskstore_epi64(outPos, _mm_cmpgt_epi64(_mm_load_si128( &inData[0] ),value), _mm_set_epi64x(0,1));
                
                if (__builtin_popcountl(*outPos)) outPos++;
                if (__builtin_popcountl(*outPos)) outPos++;
            
            for(unsigned i = 1; i < inDataCount/2; i++){
                //cmpgt -> compare
                //maskstore vector of i (mask is result of cmpgt)
                _mm_maskstore_epi64(outPos, _mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value), _mm_set_epi64x(i/2,i/2+1));
                
                if (*outPos) outPos++;
                if (*outPos) outPos++;
            }
        }
        
         if (typeid(op)==typeid(std::equal_to<uint64_t>)){
            
            _mm_maskstore_epi64(outPos, _mm_cmpeq_epi64(_mm_load_si128( &inData[0] ),value), _mm_set_epi64x(0,1));
                
                if (__builtin_popcountl(*outPos)) outPos++;
                if (__builtin_popcountl(*outPos)) outPos++;
            
            for(unsigned i = 1; i < inDataCount/2; i++){
                //cmpgt -> compare
                //maskstore vector of i (mask is result of cmpgt)
                _mm_maskstore_epi64(outPos, _mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value), _mm_set_epi64x(i/2,i/2+1));
                
                if (*outPos) outPos++;
                if (*outPos) outPos++;
            }
        }
        
        if (typeid(op)==typeid(std::greater_equal<uint64_t>)){
            
            _mm_maskstore_epi64(outPos, _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[0] ),value),_mm_cmpgt_epi64(_mm_load_si128( &inData[0] ),value)), _mm_set_epi64x(0,1));
                
                if (__builtin_popcountl(*outPos)) outPos++;
                if (__builtin_popcountl(*outPos)) outPos++;
            
            for(unsigned i = 1; i < inDataCount/2; i++){
                //cmpgt -> compare
                //maskstore vector of i (mask is result of cmpgt)
                _mm_maskstore_epi64(outPos, _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(_mm_load_si128( &inData[i] ),value)), _mm_set_epi64x(i/2,i/2+1));
                
                if (*outPos) outPos++;
                if (*outPos) outPos++;
            }
        }
        
         if (typeid(op)==typeid(std::less_equal<uint64_t>)){
            
            _mm_maskstore_epi64(outPos, _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[0] ),value),_mm_cmpgt_epi64(value,_mm_load_si128( &inData[0] ))), _mm_set_epi64x(0,1));
                
                if (__builtin_popcountl(*outPos)) outPos++;
                if (__builtin_popcountl(*outPos)) outPos++;
            
            for(unsigned i = 1; i < inDataCount/2; i++){
                //cmpgt -> compare
                //maskstore vector of i (mask is result of cmpgt)
                _mm_maskstore_epi64(outPos, _mm_or_si128(_mm_cmpeq_epi64(_mm_load_si128( &inData[i] ),value),_mm_cmpgt_epi64(value,_mm_load_si128( &inData[i] ))), _mm_set_epi64x(i/2,i/2+1));
                
                if (*outPos) outPos++;
                if (*outPos) outPos++;
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

