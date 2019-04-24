/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   manipulate.h
 * Author: Annett
 *
 * Created on 24. April 2019, 17:13
 */

#ifndef MANIPULATE_H
#define MANIPULATE_H

#include <vector/general_vector.h>

namespace vector{
    
   
   template<class VectorExtension, int IOGranularity>
   struct manipulate;

   /*!
    * Rotate vector a by 1 element (e.g. for intersection, join,...)
    */
   template<class VectorExtension, int Granularity>
   typename VectorExtension::vector_t
   rotate(typename VectorExtension::vector_t a) {
       return manipulate<VectorExtension,  Granularity>::rotate( a );
   }
}

#endif /* MANIPULATE_H */

