
#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"
// #include "../../include/core/operators/scalar/agg_sum_uncompr.h"

#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"

// #include "../../include/core/operators/general_vectorized/agg_sum_uncompr.h"
#include "../../include/core/operators/general_vectorized/agg_sum_compr.h"



#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/vector/scalar/extension_scalar.h"

#include "../../include/vector/vecprocessor/tsubasa/extension_tsubasa.h"

#include <functional>
#include <iostream>
#include <random>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    const size_t countValues = 1 * 1 * 1  * 256 + 50;
    const column<uncompr_f> * const baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(1, 50),
            true
    );
    

    std::cout << "done." << std::endl;
    std::cout.flush();
    // print_columns(print_buffer_base::decimal, baseCol1, "Test column");
    std::cout.flush();


    // ************************************************************************
    // * Query execution
    // ************************************************************************
    
    using ve = tsubasa<v16384<uint64_t> >;
    // using ve = scalar<v64<uint64_t> >;

    
    std::cout << "Query execution started... ";
    std::cout.flush();

    auto i1 = morphstore::agg_sum<ve, uncompr_f>(baseCol1);


    std::cout << "done." << std::endl << std::endl;
    
    // ************************************************************************
    // * Result output
    // ************************************************************************

    print_columns(print_buffer_base::decimal, i1, "Sum");
    
    return 0;
}