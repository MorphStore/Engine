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

   template<typename T, int IOGranularity, int Scale>
   struct gather_t <tsubasa<v16384<T>>, IOGranularity, Scale>{
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename tsubasa<v16384<U>>::vector_t
      apply(typename tsubasa<v16384<U>>::base_t const * const a, typename tsubasa<v16384<U>>::vector_t b, 
      int element_count = tsubasa<v16384<U>>::vector_helper_t::element_count::value){
         typename tsubasa<v16384<U>>::vector_t vy = _vel_vsfa_vvssl(b, 3, reinterpret_cast<uint64_t>(a), element_count); // shift left by 3 and add a, should depend from scale
         return _vel_vgt_vvssl(vy, 
                              0, //lowest address reinterpret_cast<uint64_t> (a)
                              0, //highest address
                              element_count);
      }
   };



}

#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
