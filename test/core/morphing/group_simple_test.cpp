/**
 * @file group_simple_test.cpp
 * @brief Tests of the (de)compressing morph-operators for Group-Simple, i.e.,
 * for the format `group_simple_f`.
 */

#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/group_simple.h>
#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vectorlib;


// ****************************************************************************
// Macros for the variants for variant_executor.
// ****************************************************************************

#define MAKE_VARIANT(ve, format) { \
    new typename varex_t::operator_wrapper::template for_output_formats<format>::template for_input_formats<format>( \
        &morph<ve, format, format>, true \
    ), \
    STR_EVAL_MACROS(ve), \
    STR_EVAL_MACROS(format), \
}

// ****************************************************************************
// Main program.
// ****************************************************************************

int main(void) {
    // @todo This should not be necessary.
    fail_if_self_managed_memory();
    
    using varex_t = variant_executor_helper<1, 1>::type
        ::for_variant_params<std::string, std::string>
        ::for_setting_params<unsigned, size_t>;
    varex_t varex(
            {},
            {"vector_extension", "format"},
            {"bitwidth", "countValues"}
    );
    
    std::vector<varex_t::variant_t> variants = {
        MAKE_VARIANT(scalar<v64<uint64_t>>, uncompr_f),

        MAKE_VARIANT(scalar<v64<uint64_t>>, SINGLE_ARG(group_simple_f<1, uint64_t, 8>)),
#ifdef SSE
        MAKE_VARIANT(sse<v128<uint64_t>>, SINGLE_ARG(group_simple_f<2, uint64_t, 16>)),
#endif
#ifdef AVXTWO
        MAKE_VARIANT(avx2<v256<uint64_t>>, SINGLE_ARG(group_simple_f<4, uint64_t, 32>)),
#endif
#ifdef AVX512
        MAKE_VARIANT(avx512<v512<uint64_t>>, SINGLE_ARG(group_simple_f<8, uint64_t, 64>)),
#endif
    };
    
    for(unsigned bw = 1; bw <= std::numeric_limits<uint64_t>::digits; bw++) {
        for(size_t countValues : {
            1, 2, 100, 101, 1000, 2048, 12345, 1234567
        }) {
            varex.print_datagen_started();
            auto origCol = generate_with_distr(
                    countValues,
                    std::uniform_int_distribution<uint64_t>(
                            bitwidth_min<uint64_t>(bw), bitwidth_max<uint64_t>(bw)
                    ),
                    false
            );
            varex.print_datagen_done();

            varex.execute_variants(variants, bw, countValues, origCol);

//            delete origCol;
        }
    }
    
    varex.done();
    
    return !varex.good();
}