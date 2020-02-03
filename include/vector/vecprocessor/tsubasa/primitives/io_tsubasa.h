/**
 * @file io_tsubasa.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/io.h>


namespace vectorlib {

   template<typename T, int IOGranularity>
   struct io<tsubasa<v16384<T>>, iov::ALIGNED, IOGranularity>{
      /* 64 Bit
      */
      template<typename U = T>
      MSV_CXX_ATTRIBUTE_INLINE
      static typename tsubasa<v16384<U>>::vector_t
      load( U const * const p_DataPtr , int element_count = tsubasa<v16384<U>>::vector_helper_t::element_count::value ) {
         return _vel_vld_vssl(8, reinterpret_cast< void const * >(p_DataPtr), element_count);
      }

      /* 64 Bit
      */
      template<typename U = T>
      MSV_CXX_ATTRIBUTE_INLINE
      static void
      store( U  *  p_DataPtr , typename tsubasa<v16384<U>>::vector_t p_Vec ,int element_count = tsubasa<v16384<U>>::vector_helper_t::element_count::value ) {


         return  _vel_vst_vssl(p_Vec, 8, reinterpret_cast<void*> (p_DataPtr), element_count);
      }

   };



}

#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
