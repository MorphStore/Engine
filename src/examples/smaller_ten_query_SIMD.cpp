#include "../../include/core/memory/mm_glob.h"
//#include <core/memory/mm_glob.h>

#include "../../include/core/morphing/format.h"
// #include "../../include/core/operators/scalar/agg_sum_uncompr.h"
// #include "../../include/core/operators/scalar/project_uncompr.h"
// #include "../../include/core/operators/scalar/select_uncompr.h"
#include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/operators/general_vectorized/project_uncompr.h>
#include <core/operators/general_vectorized/select_uncompr.h>

#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"

// #include "../../include/vector/scalar/extension_scalar.h"

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

// #include <core/morphing/format.h>
// #include <core/morphing/uncompr.h>

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

    const size_t countValues = 100; // generate 100 numbers
    const column<uncompr_f> * const myNumbers = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(1, 50), //range between 1 and 50
            true //numbers are sorted
    );

    std::cout << "done." << std::endl;
    print_columns(print_buffer_base::decimal, myNumbers, "myNumbers");//gives out the table

    // ************************************************************************
    // * Query execution
    // ************************************************************************

    // using ve = scalar<v64<uint64_t> >;
    // using ve = sse<v128<uint64_t>>;
    using ve = avx2<v256<uint64_t>>;


    std::cout << "Query execution started... ";
    std::cout.flush();

    // Positions fulfilling "myNumbers < 10"
    auto i1 = morphstore::select<
            less,
            ve,
            uncompr_f,
            uncompr_f
    >(myNumbers, 10);
    // Data elements of "myNumbers" fulfilling "myNumbers < 10"
    // auto i2 = project<ve, uncompr_f>(myNumbers, i1);

    std::cout << "done." << std::endl << std::endl;

    // ************************************************************************
    // * Result output
    // ************************************************************************

    // print_columns(print_buffer_base::decimal, myNumbers, "myNumbers");
    print_columns(print_buffer_base::decimal, i1, "Idx myNumbers<10");
    uint8_t *mudata = i1->get_data();
    for(int i = 0; i < 100; i++){
      std::cout << (int)mudata[i] << std::endl;
   }

    return 0;
}
