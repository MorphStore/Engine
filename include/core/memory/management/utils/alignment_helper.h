/**
 * @file alignment_helper.h
 * @brief Brief description
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_ALIGNMENT_HELPER_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_ALIGNMENT_HELPER_H

#include <core/utils/preprocessor.h>

#ifndef MSV_MEMORY_MANAGER_ALIGNMENT_BYTE
#  define MSV_MEMORY_MANAGER_ALIGNMENT_BYTE 64_B
#endif
#define MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE (MSV_MEMORY_MANAGER_ALIGNMENT_BYTE-1)
#define MSV_MEMORY_MANAGER_ALIGNMENT_TWOS_COMPLEMENT (-MSV_MEMORY_MANAGER_ALIGNMENT_BYTE)

MSV_CXX_ATTRIBUTE_FORCE_INLINE size_t get_size_with_alignment_padding( size_t p_SizeOfMemChunk ) {
   return p_SizeOfMemChunk+MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE;
}

MSV_CXX_ATTRIBUTE_FORCE_INLINE void* create_aligned_ptr(void* p_Ptr) {
   trace( "[] -  IN. ( ptr = ", p_Ptr, " )." );
   size_t ptrToSizeT = reinterpret_cast<size_t>(p_Ptr);
   size_t offset = ptrToSizeT & MSV_MEMORY_MANAGER_ALIGNMENT_MINUS_ONE_BYTE;
   trace( "[] - offset = ", offset, "." );
#if MSV_OPTIMIZE_ALIGNMENT_VARIANT==1
   if(MSV_CXX_ATTRIBUTE_LIKELY(offset)) {
      trace( "[] - OUT. ( ptr = ", reinterpret_cast<void*>(ptrToSizeT + (MSV_MEMORY_MANAGER_ALIGNMENT_BYTE - offset)), " ).");
      return reinterpret_cast<void*>(ptrToSizeT + (MSV_MEMORY_MANAGER_ALIGNMENT_BYTE - offset));
   }
#elif MSV_OPTIMIZE_ALIGNMENT_VARIANT==2
   trace( "[] - OUT. ( ptr = ", reinterpret_cast<void*>(ptrToSizeT + ((offset!=0) * (MSV_MEMORY_MANAGER_ALIGNMENT_BYTE - offset))), " ).");
   return reinterpret_cast<void*>( ptrToSizeT + ( (offset!=0) * (MSV_MEMORY_MANAGER_ALIGNMENT_BYTE - offset) ) );
#else
   trace(
      "[] - OUT. ( ptr = ",
      reinterpret_cast<void*>(ptrToSizeT + MSV_MEMORY_MANAGER_ALIGNMENT_BYTE +
         (-offset | MSV_MEMORY_MANAGER_ALIGNMENT_TWOS_COMPLEMENT)),
      ". padding = ", MSV_MEMORY_MANAGER_ALIGNMENT_BYTE + (-offset | MSV_MEMORY_MANAGER_ALIGNMENT_TWOS_COMPLEMENT), " ).");
   return reinterpret_cast<void*>( ptrToSizeT + MSV_MEMORY_MANAGER_ALIGNMENT_BYTE + (-offset | MSV_MEMORY_MANAGER_ALIGNMENT_TWOS_COMPLEMENT) );
#endif
}

MSV_CXX_ATTRIBUTE_FORCE_INLINE void* create_extended_aligned_ptr(void* p_StartPtr, size_t p_SizeOfMemChunk) {
   trace( "[] -  IN. ( ptr = ", p_StartPtr, ". Size = ", p_SizeOfMemChunk, " Bytes )." );
   void * aligned_ptr = create_aligned_ptr(p_StartPtr);
   (reinterpret_cast<size_t *>(aligned_ptr))[-1] = p_SizeOfMemChunk;
   trace( "[] - OUT. ( ptr = ", aligned_ptr, ". ptr[-1] = ", reinterpret_cast<size_t*>(aligned_ptr)[-1], " )." );
   return aligned_ptr;
}


#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_ALIGNMENT_HELPER_H
