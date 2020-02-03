/**
 * @file io_tsubasa.h
 * @brief Brief description
 * @author 
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CREATE_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CREATE_TSUBASA_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/create.h>

namespace vectorlib {
    template<typename T>
    struct create <tsubasa<v16384<T>>, 64> {
        template<typename U = T>
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename tsubasa< v16384< U > >::vector_t
        set1(uint64_t number, int element_count = tsubasa<v16384<U>>::vector_helper_t::element_count::value){
            return _vel_pvbrd_vsl(number, element_count);
        }
    };
}


#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_IO_TSUBASA_H
