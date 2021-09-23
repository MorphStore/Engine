#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"
#include "../../include/core/operators/scalar/agg_sum_uncompr.h"
#include "../../include/core/operators/scalar/group_uncompr.h"
#include "../../include/core/operators/scalar/join_uncompr.h"
#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"
#include "../../include/core/operators/scalar/intersect_uncompr.h"
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


template<class T>
struct equals {
    constexpr bool operator()(const T &lhs, const T &rhs) const {
        return lhs == rhs;
    }
};


int main(void) {

    // *****************************************************************************************************************
    // * Query
    // *****************************************************************************************************************


    std::cout << "Provider_8" << std::endl;

    // SELECT "Provider_8"."nppes_provider_city" AS "nppes_provider_city" FROM "Provider_8" WHERE
    // (("Provider_8"."nppes_provider_state" = \'WA\')
    // AND ("Provider_8"."provider_type" = \'Diagnostic Radiology\'))
    // GROUP BY "Provider_8"."nppes_provider_city";

    // TRANSLATION

    // SELECT "Provider_8"."nppes_provider_city" AS "nppes_provider_city" FROM "Provider_8" WHERE
    // (("Provider_8"."nppes_provider_state" = 55) AND ("Provider_8"."provider_type" = 20)) GROUP BY
    // "Provider_8"."nppes_provider_city";



    // *****************************************************************************************************************
    // * Column generation
    // *****************************************************************************************************************


    auto nppes_provider_city = binary_io<uncompr_f>::load("public_bi_data/provider/13.bin");
    auto nppes_provider_state = binary_io<uncompr_f>::load("public_bi_data/provider/19.bin");
    auto provider_type = binary_io<uncompr_f>::load("public_bi_data/provider/24.bin");



    // *****************************************************************************************************************
    // * Query execution
    // *****************************************************************************************************************


    using ps = scalar<v64 < uint64_t>>;

    auto intermediate_1 = select<equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 55);
    auto intermediate_2 = select<equals, ps, uncompr_f, uncompr_f>(provider_type, 20);
    auto intermediate_3 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_1, intermediate_2);
    auto intermediate_4 = project<ps, uncompr_f>(nppes_provider_city, intermediate_3);
    auto grouped = group<ps, uncompr_f, uncompr_f>(intermediate_4);
    auto groups = std::get<1>(grouped);
    auto output = project<ps, uncompr_f>(intermediate_4, groups);


    // *****************************************************************************************************************
    // * Result output
    // *****************************************************************************************************************


    print_columns(print_buffer_base::decimal, output, "nppes_provider_city");


    return 0;
}
