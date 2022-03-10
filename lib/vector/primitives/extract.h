/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   extract.h
 * Author: Annett
 *
 * Created on 25. April 2019, 13:57
 */

#ifndef EXTRACT_H
#define EXTRACT_H

#include "../vector_extension_structs.h"

namespace vectorlib{


   template<class VectorExtension, int IOGranularity>
   struct extract;


    /*! Extract a value from a vector
     * @param a the vector to extract the value from
     * @param b index
    */
   #ifdef __CUDACC__
   template<class VectorExtension,  int IOGranularity>
   typename VectorExtension::base_t*
   extract_value(typename VectorExtension::vector_t a, int b){
      return extract<VectorExtension, IOGranularity>::extract_value(a, b);
   }
  #else
   template<class VectorExtension,  int IOGranularity>
   typename VectorExtension::base_t
   extract_value(typename VectorExtension::vector_t a, int b){
      return extract<VectorExtension, IOGranularity>::extract_value(a, b);
   }
  #endif

}

#endif /* EXTRACT_H */

