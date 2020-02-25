#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"

// #include "../../include/core/operators/scalar/project_uncompr.h"

// #include "../../include/core/operators/general_vectorized/agg_sum_uncompr.h"
#include "../../include/core/operators/general_vectorized/agg_sum_compr.h"
#include "../../include/core/operators/general_vectorized/project_compr.h"

#include "../../include/core/operators/interfaces/project.h"


#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/vector/scalar/extension_scalar.h"

#include "../../include/vector/vecprocessor/tsubasa/extension_tsubasa.h"

#include <functional>
#include <iostream>
#include <random>

#include <fstream>
#include <vector>

#include <math.h>

#include <chrono>

#include "time_tsubasa.h"

using namespace morphstore;
using namespace vectorlib;

int main (void) {

    using ve = tsubasa<v16384<uint64_t>>;

    const size_t countValues = 1000;
    const size_t posValues = 260;



    const column<uncompr_f> *  baseCol = generate_with_distr(
    countValues,
    std::uniform_int_distribution<uint64_t>(1, 50),
    true
    );

    const column<uncompr_f> *  posCol = generate_sorted_unique_extraction(posValues, countValues);

    auto resCol = morphstore::my_project_wit_t<ve, uncompr_f, uncompr_f, uncompr_f>::apply(baseCol, posCol);


    print_columns(print_buffer_base::decimal, baseCol, "BaseCol");

    print_columns(print_buffer_base::decimal, posCol, "posCol");



    print_columns(print_buffer_base::decimal, resCol, "Result");



    return 0;
}