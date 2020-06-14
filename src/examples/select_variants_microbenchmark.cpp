/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file select_variants_microbenchmark.cpp
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/vbp.h>
#include <core/morphing/vbp_padding.h>
#include <core/morphing/type_packing.h>
#include <core/operators/general_vectorized/select_compr.h>
//#include <core/operators/general_vectorized/select_uncompr.h>
#include <core/operators/specialized/select_type_packing.h>
#include <core/operators/specialized/select_type_packing_32to32.h>
#include <core/operators/specialized/select_type_packing_32to64.h>
#include <core/operators/specialized/select_type_packing_64to32.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;

#define MAKE_VARIANT(op, cmp, vector_extension, out_pos_f, in_data_f) { \
    new typename varex_t::operator_wrapper::template for_output_formats<out_pos_f>::template for_input_formats<in_data_f>( \
        &op<cmp, vector_extension, out_pos_f, in_data_f>::apply \
    ), \
    STR_EVAL_MACROS(vector_extension), \
    STR_EVAL_MACROS(out_pos_f), \
    STR_EVAL_MACROS(in_data_f), \
}


// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1, uint64_t, size_t>::type
        ::for_variant_params<std::string, std::string, std::string>
        ::for_setting_params<float>;
    varex_t varex(
            {"pred", "est"},
            {"vector_extension", "out_pos_f", "in_data_f"},
            {"selectivity"}
    );
    
    const uint64_t pred = 0;
    
    const size_t countValues = 10000000;///799;//10000000;//18;;//17;//50000007;

    std::vector<varex_t::variant_t> variants = {
        //MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<32, 2> >)),          
        MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, uncompr_f, uncompr_f),
        MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, uncompr_f, SINGLE_ARG(static_vbp_f<vbp_l<32, 2> >)),
        // MAKE_VARIANT(my_select_wit_t, equal, avx2<v256<uint64_t>>, uncompr_f, uncompr_f),
        // MAKE_VARIANT(my_select_wit_t, equal, avx512<v512<uint64_t>>, uncompr_f, uncompr_f),

        // MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint64_t >)), 
        // MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint32_t >)),
        // MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint64_t >)),
        // MAKE_VARIANT(my_select_wit_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint32_t >)),   

        // MAKE_VARIANT(my_select_wit_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint64_t >)), 
        // MAKE_VARIANT(my_select_wit_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint32_t >)),
        // MAKE_VARIANT(my_select_wit_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint64_t >)),
        // MAKE_VARIANT(my_select_wit_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint32_t >)),   

        // MAKE_VARIANT(my_select_wit_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint64_t >)),                            
        // MAKE_VARIANT(my_select_wit_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint32_t >)),
        // MAKE_VARIANT(my_select_wit_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint64_t >)),
        // MAKE_VARIANT(my_select_wit_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint32_t >))  


        MAKE_VARIANT(select_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint64_t >)), 
        MAKE_VARIANT(select_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint32_t >)),
        MAKE_VARIANT(select_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint64_t >)),
        MAKE_VARIANT(select_t, equal, sse<v128<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint32_t >)),   

        MAKE_VARIANT(select_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint64_t >)), 
        MAKE_VARIANT(select_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint32_t >)),
        MAKE_VARIANT(select_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint64_t >)),
        MAKE_VARIANT(select_t, equal, avx2<v256<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint32_t >)),   

        MAKE_VARIANT(select_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint64_t >)),                            
        MAKE_VARIANT(select_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint64_t >), SINGLE_ARG(type_packing_f<uint32_t >)),
        MAKE_VARIANT(select_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint64_t >)),
        MAKE_VARIANT(select_t, equal, avx512<v512<uint64_t>>, SINGLE_ARG(type_packing_f<uint32_t >), SINGLE_ARG(type_packing_f<uint32_t >))   

    };
    
    for(float selectivity : {
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    }) {
        varex.print_datagen_started();
        const size_t countMatches = static_cast<size_t>(
                ((static_cast<float>(countValues) * selectivity))
        );
        auto origCol = generate_exact_number(
                countValues,
                countMatches,
                pred,
                bitwidth_max<uint64_t>(16)
        );
        varex.print_datagen_done();

        varex.execute_variants(
                variants, selectivity, origCol, pred, 0
        );

        delete origCol;
    }
    
    varex.done();
    
    return 0;
}

