#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"
#include "../../include/core/operators/scalar/agg_sum_uncompr.h"
#include "../../include/core/operators/scalar/group_uncompr.h"
#include "../../include/core/operators/scalar/join_uncompr.h"
#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"
#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/vector/scalar/extension_scalar.h"
#include "../../include/core/persistence/binary_io.h"

#include <functional>
#include <iostream>
#include <random>
#include <tuple>

using namespace morphstore;
using namespace vectorlib;

int main() {

    // *****************************************************************************************************************
    // * Query
    // *****************************************************************************************************************


    std::cout << "CityMaxCapita_1" << std::endl;

    // SELECT "CityMaxCapita_1"."City" AS "City",
    // SUM(1) AS "TEMP(Calculation_383650433986449409)(2109769841)(0)"
    // FROM "CityMaxCapita_1" GROUP BY "CityMaxCapita_1"."City";



    // *****************************************************************************************************************
    // * Column generation
    // *****************************************************************************************************************


    auto cmc_city = binary_io<uncompr_f>::load("public_bi_data/citymaxcapita/2.bin");
    auto cmc_const = binary_io<uncompr_f>::load("public_bi_data/citymaxcapita/const.bin");





    // *****************************************************************************************************************
    // * Query execution
    // *****************************************************************************************************************


    using ps = scalar<v64 < uint64_t>>;

    auto grouped = group<ps, uncompr_f, uncompr_f>(cmc_city);
    auto group_ids = std::get<0>(grouped);
    auto groups = std::get<1>(grouped);
    auto i1 = project<ps, uncompr_f>(cmc_city, groups);
    auto i2 = agg_sum<ps, uncompr_f>(group_ids, cmc_const, groups->get_count_values());



    // *****************************************************************************************************************
    // * Result output
    // *****************************************************************************************************************


    print_columns(print_buffer_base::decimal, i1, i2, "City", "TEMP(Calculation_383650433986449409)(2109769841)(0)");



    return 0;
}

