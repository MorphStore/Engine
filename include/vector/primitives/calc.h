/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   calc.h
 * Author: Annett
 *
 * Created on 17. April 2019, 11:07
 */

#ifndef CALC_H
#define CALC_H

#include <vector/general_vector.h>

namespace vector{
    
   
   template<class VectorExtension, int IOGranularity>
   struct calc;

   /*!
    * Add Vectors a and b component wise. Granularity gives the size of a component in bit, 
    * e.g. to add 64-bit integers, granularity is 64. To add float values, Granularity is 32.
    */
   template<class VectorExtension, int Granularity>
   typename VectorExtension::vector_t
   add(typename VectorExtension::vector_t a, typename VectorExtension::vector_t b ) {
       return calc<VectorExtension,  Granularity>::add( a, b );
   }
   
    /*!
    * Subtract Vector b from a component wise. Granularity gives the size of a component in bit, 
    * e.g. to subtract 64-bit integers, Granularity is 64. To subtract float values, Granularity is 32.
    */
   
   template<class VectorExtension, int Granularity>
   typename VectorExtension::vector_t
   sub(typename VectorExtension::vector_t a, typename VectorExtension::vector_t b ) {
       return calc<VectorExtension,  Granularity>::sub( a, b );
   }
   
    /*!
    * Builds the sum of all elements in vector a. This is really ugly if done 
    * vectorized using sse or avx2 (sequentially might be faster in these cases).
    */
   
   template<class VectorExtension, int Granularity>
   typename VectorExtension::base_t
   hadd(typename VectorExtension::vector_t a) {
       return calc<VectorExtension,  Granularity>::hadd( a );
   }
   
}
#endif /* CALC_H */

