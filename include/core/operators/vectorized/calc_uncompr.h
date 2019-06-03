/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc_uncompr.h
 * Author: Annett
 *
 * Created on 9. April 2019, 17:55
 */

#ifndef CALC_UNCOMPR_H
#define CALC_UNCOMPR_H


#include <core/operators/interfaces/calc.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/processing_style.h>

#include <cstdint>
#include <stdexcept>
#include <immintrin.h>

/*! Known issues:
 *  1. Multiplication/Division/Modulo work only for lower 32/52 bit of each 64-bit value
 *
 */
namespace morphstore {
    
template<template<typename> class t_unary_op>
struct calc_unary<
        t_unary_op,
        processing_style_t::vec256,
        uncompr_f,
        uncompr_f
> {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataCol
    ) {
        const size_t inDataCount = inDataCol->get_count_values();
        const __m256i * const inData = (__m256i*)inDataCol->get_data();
        
        const size_t inDataSize = inDataCol->get_size_used_byte();
        // Exact allocation size (for uncompressed data).
        auto outDataCol = new column<uncompr_f>(inDataSize);
        __m256i * outData = outDataCol->get_data();
        const uint64_t * outDataInit = (uint64_t *) outData;
        
        t_unary_op<uint64_t> op;
        __m256i zeros=_mm256_set1_epi64x(0);
        
        if (typeid(op)==typeid(std::negate<uint64_t>)){
            for(unsigned i = 0; i < inDataCount/4; i++){
                _mm256_store_si256(outData,_mm256_sub_epi64(zeros,_mm256_load_si256(inData+i)));
                outData++;
            }
        }
           
         //Process the last elements (which do not fill up a whole register) sequentially 
        uint64_t * oData=(uint64_t *)outData;
        for(unsigned i = (outDataInit+inDataSize-(uint64_t*)outData); i < inDataCount; i++)
            oData[i] = op(((uint64_t*)inData)[i]);
        
        outDataCol->set_meta_data(inDataCount, inDataSize);
        
        return outDataCol;
    }
};
    
template<template<typename> class t_binary_op>
struct calc_binary<
        t_binary_op, processing_style_t::vec256,
        uncompr_f,
        uncompr_f,
        uncompr_f
> {
    static
    const column<uncompr_f> * apply(
            const column<uncompr_f> * const inDataLCol,
            const column<uncompr_f> * const inDataRCol
    ) {
        const size_t inDataCount = inDataLCol->get_count_values();
    
        if(inDataCount != inDataRCol->get_count_values())
            throw std::runtime_error(
                    "calc: inDataLCol and inDataRCol must contain the same "
                    "number of data elements"
            );
        
        const __m256i * const inDataL = inDataLCol->get_data();
        const __m256i * const inDataR = inDataRCol->get_data();
        
        const size_t inDataSize = inDataLCol->get_size_used_byte();
        // Exact allocation size (for uncompressed data).
        auto outDataCol = new column<uncompr_f>(inDataSize);
        __m256i * outData = (__m256i*)outDataCol->get_data();
        //const uint64_t * outDataInit = (uint64_t *) outData;
        
        t_binary_op<uint64_t> op;
        
        //Some helper registers, we will need for casting magic in the division 
        __m256d divhelper=_mm256_set1_pd(0x0010000000000000);
        __m256d intermediate;
        unsigned i=0;
        if (typeid(op)==typeid(std::plus<uint64_t>)){
            for( i = 0; i < inDataCount/4; i++){
                _mm256_store_si256(outData,_mm256_add_epi64(_mm256_load_si256(inDataL+i),_mm256_load_si256(inDataR+i)));
                outData++;
            }
        }
        
        if (typeid(op)==typeid(std::minus<uint64_t>)){
            for( i = 0; i < inDataCount/4; i++){
              _mm256_store_si256(outData,_mm256_sub_epi64(_mm256_load_si256(inDataL+i),_mm256_load_si256(inDataR+i)));
              outData++;
            }
        }

        if (typeid(op)==typeid(std::multiplies<uint64_t>)){
            for( i = 0; i < inDataCount/4; i++){
              _mm256_store_si256(outData,_mm256_mul_epu32(_mm256_load_si256(inDataL+i),_mm256_load_si256(inDataR+i)));
              outData++;
            }
        }        
        

         if (typeid(op)==typeid(std::divides<uint64_t>)){
            for( i = 0; i < inDataCount/4; i++){
                
                //Load as double and divide -> 64-bit integer division is not supported in sse or avx(2) 
                intermediate=_mm256_div_pd(_mm256_load_pd((const double*)(inDataL+i)),_mm256_load_pd((const double *)(inDataR+i)));
                
                //Make an integer out of the double by adding a bit at the 52nd position and XORing the result bitwise with the bit at the 52nd position (all other bits are 0)
                intermediate=_mm256_add_pd(intermediate,divhelper);
                
                //store the result
                _mm256_store_si256(outData,_mm256_xor_si256(_mm256_castpd_si256(intermediate),_mm256_castpd_si256(divhelper)));
                
                outData++;
              
            }
        }
        
        if (typeid(op)==typeid(std::modulus<uint64_t>)){
            __m256i mod_intermediate;
            __m256d left;
            __m256i lefti;
            __m256d right;
            __m256i righti;
            
            for( i = 0; i < inDataCount/4; i++){
               
                // approach: divide -> floor -> multiply again -> difference between result and original
                
                //Load as double
                left=_mm256_load_pd((const double*)(inDataL+i));
                lefti=_mm256_load_si256((inDataL+i));
                //left=_mm256_set1_pd(100.0f);
                right=_mm256_load_pd((const double *)(inDataR+i));
                righti=_mm256_load_si256((inDataR+i));
                //Divide -> 64-bit integer division is not supported in sse or avx(2) 
                intermediate=_mm256_div_pd(left,right);
                
                //Floor
                intermediate=_mm256_floor_pd(intermediate);
                
                //Make an integer out of the double by adding a bit at the 52nd position and masking out mantisse and sign
                intermediate=_mm256_add_pd(intermediate,divhelper);
                mod_intermediate=_mm256_xor_si256(_mm256_castpd_si256(intermediate),_mm256_castpd_si256(divhelper));
                
                //multiply again
                mod_intermediate=_mm256_mul_epi32(mod_intermediate,righti);
                
                //difference beween result and original
                mod_intermediate=_mm256_sub_epi64(lefti,mod_intermediate);
                             
                //store the result
                _mm256_store_si256(outData,mod_intermediate);
                
                outData++;
                
            }
        }
        
        //Process the last elements (which do not fill up a whole register) sequentially 
        uint64_t * oData=(uint64_t *)outData;
        for(unsigned j = i*4; j < inDataCount; j++){
            oData[0] = (uint64_t) op(((uint64_t*)inDataL)[j],((uint64_t*)inDataR)[j]);
            oData++;
        }
        
              
        outDataCol->set_meta_data(inDataCount, inDataSize);
        
        return outDataCol;
    }
};

}

#endif /* CALC_UNCOMPR_H */

