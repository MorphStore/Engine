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
#include "../../include/vector/scalar/primitives/calc_scalar.h"
#include "../../include/core/persistence/binary_io.h"
#include "../../include/core/operators/general_vectorized/group_binary_uncompr.h"
#include "../../include/core/operators/general_vectorized/calc_uncompr.h"

#include <functional>
#include <iostream>
#include <random>
#include <tuple>

using namespace morphstore;
using namespace vectorlib;


template<class T >
struct not_equals {
    constexpr bool operator()(const T &lhs, const T &rhs) const
    {
        return lhs != rhs;
    }
};


int main() {

    // *****************************************************************************************************************
    // * Query
    // *****************************************************************************************************************


    std::cout << "Medicare1_2" << std::endl;

    // SELECT "Medicare1_2"."NPPES_PROVIDER_CITY" AS "NPPES_PROVIDER_CITY",
    // "Medicare1_2"."NPPES_PROVIDER_STATE" AS "NPPES_PROVIDER_STATE",
    // AVG(CAST("Medicare1_2"."Calculation_9030826185528129" AS double)) AS "avg:Calculation_9030826185528129:ok"
    // FROM "Medicare1_2"
    // WHERE (NOT ("Medicare1_2"."NPPES_PROVIDER_STATE" IN ('AK', 'AS', 'GU', 'HI', 'MP', 'PR', 'VI')))
    // GROUP BY "Medicare1_2"."NPPES_PROVIDER_CITY",   "Medicare1_2"."NPPES_PROVIDER_STATE";

    // TRANSLATION

    // SELECT "Medicare1_2"."NPPES_PROVIDER_CITY" AS "NPPES_PROVIDER_CITY",
    // "Medicare1_2"."NPPES_PROVIDER_STATE" AS "NPPES_PROVIDER_STATE",
    // AVG("Medicare1_2"."Calculation_9030826185528129") AS "avg:Calculation_9030826185528129:ok"
    // FROM "Medicare1_2" WHERE ("Medicare1_2"."NPPES_PROVIDER_STATE" != 2
    // AND "Medicare1_2"."NPPES_PROVIDER_STATE" != 6
    // AND "Medicare1_2"."NPPES_PROVIDER_STATE" != 15
    // AND "Medicare1_2"."NPPES_PROVIDER_STATE" != 16
    // AND "Medicare1_2"."NPPES_PROVIDER_STATE" != 30
    // AND "Medicare1_2"."NPPES_PROVIDER_STATE" != 45
    // AND "Medicare1_2"."NPPES_PROVIDER_STATE" != 53)
    // GROUP BY "Medicare1_2"."NPPES_PROVIDER_CITY",   "Medicare1_2"."NPPES_PROVIDER_STATE";



    // *****************************************************************************************************************
    // * Column generation
    // *****************************************************************************************************************


    auto nppes_provider_city = binary_io<uncompr_f>::load("public_bi_data/medicare/12.bin");
    auto nppes_provider_state = binary_io<uncompr_f>::load("public_bi_data/medicare/15.bin");
    auto calculation_9030826185528129 = binary_io<uncompr_f>::load("public_bi_data/medicare/6.bin");



    // *****************************************************************************************************************
    // * Query execution
    // *****************************************************************************************************************


    using ps = scalar<v64<uint64_t>>;

    auto intermediate_1 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 2);
    auto intermediate_2 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 6);
    auto intermediate_3 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 15);
    auto intermediate_4 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 16);
    auto intermediate_5 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 30);
    auto intermediate_6 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 45);
    auto intermediate_7 = select<not_equals, ps, uncompr_f, uncompr_f>(nppes_provider_state, 53);

    auto intermediate_8 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_1, intermediate_2);
    auto intermediate_9 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_8, intermediate_3);
    auto intermediate_10 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_9, intermediate_4);
    auto intermediate_11 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_10, intermediate_5);
    auto intermediate_12 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_11, intermediate_6);
    auto intermediate_13 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_12, intermediate_7);

    auto intermediate_14 = project<ps, uncompr_f>(nppes_provider_city, intermediate_13);
    auto intermediate_15 = project<ps, uncompr_f>(nppes_provider_state, intermediate_13);
    auto intermediate_16 = project<ps, uncompr_f>(calculation_9030826185528129, intermediate_13);

    auto grouped_1 = group<ps, uncompr_f, uncompr_f>(intermediate_14);
    auto group_ids_1 = std::get<0>(grouped_1);


    auto count_col = binary_io<uncompr_f>::load("public_bi_data/medicare/const_2.bin");

    auto grouped_2 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(group_ids_1, intermediate_15);
    auto group_ids_2 = std::get<0>(grouped_2);
    auto groups_2 = std::get<1>(grouped_2);

    auto intermediate_17 = agg_sum<ps, uncompr_f>(group_ids_2, intermediate_16, groups_2->get_count_values());
    auto intermediate_17_1 = agg_sum<ps, uncompr_f>(group_ids_2, count_col, groups_2->get_count_values());
    auto intermediate_18 = calc_binary<morphstore::div, ps, uncompr_f, uncompr_f, uncompr_f>(intermediate_17, intermediate_17_1);

    auto output_1 = project<ps, uncompr_f, uncompr_f>(intermediate_14, groups_2);
    auto output_2 = project<ps, uncompr_f, uncompr_f>(intermediate_15, groups_2);






    // *****************************************************************************************************************
    // * Result output
    // *****************************************************************************************************************


    print_columns(print_buffer_base::decimal, output_1, output_2, intermediate_18,
                  "NPPES_PROVIDER_CITY", "NPPES_PROVIDER_STATE", "avg:Calculation_9030826185528129:ok");


    return 0;
}

