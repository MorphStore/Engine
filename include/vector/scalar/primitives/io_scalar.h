/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   io.h
 * Author: Annett
 *
 * Created on 21. Mai 2019, 12:17
 */

#ifndef IO_SCALAR_H
#define IO_SCALAR_H


#include <core/utils/logger.h>
#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/primitives/io.h>

#include <functional>

namespace vectorlib {
    
    
   template<typename T, int IOGranularity>
   struct io<scalar<v64<T>>,iov::ALIGNED, IOGranularity> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned integer values into 256 Bit vector register." );
         return *p_DataPtr;
      }
      
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U *& p_DataPtr, typename vectorlib::scalar<v64< U > >::vector_t & p_vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         *p_DataPtr=p_vec;
         return;
      }



            
     
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading aligned double values into 256 Bit vector register." );
         return *p_DataPtr;
      }
      
      template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, vectorlib::scalar<v64< double > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned double values to memory" );
         *p_DataPtr=p_vec;
         return;
      }

   };
   
   template<typename T, int IOGranularity, int Scale>
   struct gather_t<scalar<v64<T>>, IOGranularity, Scale> {
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar< v64< U > >::vector_t
      apply( U const * const p_DataPtr, typename vectorlib::scalar<v64< U > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         return *reinterpret_cast<U const *>(reinterpret_cast<uint8_t const *>(p_DataPtr) + p_vec * Scale);
         
      }
       
   };
   
   template<typename T, int IOGranularity>
   struct io<scalar<v64<T>>,iov::UNALIGNED, IOGranularity> {

        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static typename scalar<v64< U > >::vector_t
      load( U const * const p_DataPtr ) {
         trace( "[VECTOR] - Loading unaligned double value (scalar)" );
         return *p_DataPtr;
      }
             
       template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      compressstore( U * p_DataPtr,  typename scalar< v64< U > >::vector_t p_vec, int mask ) {
         trace( "[VECTOR] - Store masked unaligned integer values to memory" );
   
         if (mask!=0)  *p_DataPtr=p_vec;
        
         return ;
      }
       
      template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
      MSV_CXX_ATTRIBUTE_FORCE_INLINE
      static void
      store( U * p_DataPtr, typename vectorlib::scalar<v64< U > >::vector_t p_vec ) {
         trace( "[VECTOR] - Store aligned integer values to memory" );
         *p_DataPtr=p_vec;
         return;
      }
       
   };
}


#endif /* IO_SCALAR_H */

