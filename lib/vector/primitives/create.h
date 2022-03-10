/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   create.h
 * Author: Annett
 *
 * Created on 25. April 2019, 10:55
 */

#ifndef CREATE_H
#define CREATE_H

#include "../vector_extension_structs.h"

namespace vectorlib{


   template<class VectorExtension, int IOGranularity>
   struct create;


    /*! Create a new vector by broadcasting value a to all elements of a vector.
    * This is a wrapper for _mm*_set1_*
    */
   template<class VectorExtension,  int IOGranularity>
   typename VectorExtension::vector_t
   set1(typename VectorExtension::base_t a){
      return create<VectorExtension, IOGranularity>::set1( a);
   }

    /*! Creates a vector filled with sequential numbers starting at a with step width b.
     * @todo: should be typename VectorExtension::base_t instead of int
    */
   template<class VectorExtension,  int IOGranularity>
   typename VectorExtension::vector_t
   set_sequence(int a, int b){
      return create<VectorExtension, IOGranularity>::set_sequence( a, b);
   }

    /*! Create a new vector by broadcasting value a to all elements of a vector.
    * This is a wrapper for _mm*_set1_*
    *
    */
   template<class VectorExtension, int IOGranularity>
   typename VectorExtension::vector_t
   set(typename VectorExtension::base_t a31, typename VectorExtension::base_t a30, typename VectorExtension::base_t a29, typename VectorExtension::base_t a28, typename VectorExtension::base_t a27,
    typename VectorExtension::base_t a26, typename VectorExtension::base_t a25, typename VectorExtension::base_t a24, typename VectorExtension::base_t a23, typename VectorExtension::base_t a22,
    typename VectorExtension::base_t a21, typename VectorExtension::base_t a20, typename VectorExtension::base_t a19, typename VectorExtension::base_t a18, typename VectorExtension::base_t a17, typename VectorExtension::base_t a16,
    typename VectorExtension::base_t a15,typename VectorExtension::base_t a14,typename VectorExtension::base_t a13,typename VectorExtension::base_t a12,typename VectorExtension::base_t a11,
           typename VectorExtension::base_t a10,typename VectorExtension::base_t a9,typename VectorExtension::base_t a8,typename VectorExtension::base_t a7,typename VectorExtension::base_t a6,
           typename VectorExtension::base_t a5,typename VectorExtension::base_t a4,typename VectorExtension::base_t a3,typename VectorExtension::base_t a2,typename VectorExtension::base_t a1,
           typename VectorExtension::base_t a0 ){
      return create<VectorExtension,  IOGranularity>::set(a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16, a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0);
   }

   template<class VectorExtension, int IOGranularity>
   typename VectorExtension::vector_t
   set(typename VectorExtension::base_t a15,typename VectorExtension::base_t a14,typename VectorExtension::base_t a13,typename VectorExtension::base_t a12,typename VectorExtension::base_t a11,
           typename VectorExtension::base_t a10,typename VectorExtension::base_t a9,typename VectorExtension::base_t a8,typename VectorExtension::base_t a7,typename VectorExtension::base_t a6,
           typename VectorExtension::base_t a5,typename VectorExtension::base_t a4,typename VectorExtension::base_t a3,typename VectorExtension::base_t a2,typename VectorExtension::base_t a1,
           typename VectorExtension::base_t a0 ){
      return create<VectorExtension,  IOGranularity>::set( a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0);
   }

   template<class VectorExtension, int IOGranularity>
   typename VectorExtension::vector_t
   set(typename VectorExtension::base_t a7,typename VectorExtension::base_t a6,typename VectorExtension::base_t a5,typename VectorExtension::base_t a4,typename VectorExtension::base_t a3,
           typename VectorExtension::base_t a2,typename VectorExtension::base_t a1,typename VectorExtension::base_t a0){
      return create<VectorExtension,  IOGranularity>::set( a7, a6, a5, a4, a3, a2, a1, a0);
   }

   template<class VectorExtension, int IOGranularity>
   typename VectorExtension::vector_t
   set(typename VectorExtension::base_t a3,typename VectorExtension::base_t a2,typename VectorExtension::base_t a1,typename VectorExtension::base_t a0){
      return create<VectorExtension,  IOGranularity>::set( a3, a2, a1, a0);
   }

   template<class VectorExtension, int IOGranularity>
   typename VectorExtension::vector_t
   set(typename VectorExtension::base_t a1,typename VectorExtension::base_t a0){
      return create<VectorExtension,  IOGranularity>::set( a1, a0);
   }
}

#endif /* CREATE_H */

