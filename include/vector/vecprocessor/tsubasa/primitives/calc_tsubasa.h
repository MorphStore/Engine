/**
 * @file io_tsubasa.h
 * @brief Brief description
 * @author 
 * @todo TODOS?
 */

#ifndef MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H
#define MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H


#include <core/utils/preprocessor.h>
#include <core/memory/mm_glob.h>
#include <vector/vecprocessor/tsubasa/extension_tsubasa.h>
#include <vector/primitives/calc.h>

// #include <iostream>

namespace vectorlib {
    template<>
    struct add <tsubasa<v16384<uint64_t>>> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename tsubasa< v16384< uint64_t > >::vector_t
        apply(      
            typename tsubasa<v16384<uint64_t>>::vector_t const & p_vec1,
            typename tsubasa<v16384<uint64_t>>::vector_t const & p_vec2,
            int element_count = tsubasa<v16384<uint64_t>>::vector_helper_t::element_count::value
            )
            {
            // std::cout << "Addition mit: " << element_count << "Elementen" << std::endl;
            return _vel_vaddul_vvvl(p_vec1, p_vec2, element_count);
        }
    };

    template<>
    struct hadd <tsubasa<v16384<uint64_t>>> {
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static typename tsubasa< v16384< uint64_t > >::base_t
        apply(      
            typename tsubasa<v16384<uint64_t>>::vector_t const & p_vec1,
            int element_count = tsubasa<v16384<uint64_t>>::vector_helper_t::element_count::value
            )
            {
            return _vel_lvsl_svs(_vel_vsuml_vvl(p_vec1, element_count), 0);;
        }
    };
}


#endif //MORPHSTORE_VECTOR_VECPROCESSOR_TSUBASA_PRIMITIVES_CALC_TSUBASA_H