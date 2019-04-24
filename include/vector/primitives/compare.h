/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   compare.h
 * Author: Annett
 *
 * Created on 23. April 2019, 16:53
 */

#ifndef COMPARE_H
#define COMPARE_H

#include <vector/general_vector.h>

namespace vector{
    
   
   template<class VectorExtension, int IOGranularity>
   struct compare;

   /*!
    * Compares two vectors element wise for equality
    */
   template<class VectorExtension, int Granularity>
   int
   equality(typename VectorExtension::vector_t a, typename VectorExtension::vector_t b ) {
       return compare<VectorExtension,  Granularity>::equality( a, b );
   }
}

#endif /* COMPARE_H */

