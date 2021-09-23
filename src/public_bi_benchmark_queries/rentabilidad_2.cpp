#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"
#include "../../include/core/operators/scalar/agg_sum_uncompr.h"
//#include "../../include/core/operators/uncompr/agg_sum_all.h"
//#include "../../include/core/operators/uncompr/agg_sum_grouped.h"
//#include "../../include/core/operators/uncompr/group_first.h"
//#include "../../include/core/operators/uncompr/project.h"
#include "../../include/core/operators/scalar/group_uncompr.h"
#include "../../include/core/operators/scalar/join_uncompr.h"
#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"
#include "../../include/core/operators/scalar/intersect_uncompr.h"
#include "../../include/core/operators/scalar/calc_uncompr.h"
#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/vector/scalar/extension_scalar.h"
#include "../../include/vector/scalar/primitives/calc_scalar.h"
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


int main() {

    // *****************************************************************************************************************
    // * Query
    // *****************************************************************************************************************


    std::cout << "Rentabilidad_1_1" << std::endl;

    //SELECT "Rentabilidad_1"."GEC (group)" AS "GEC (group)",   SUM("Rentabilidad_1"."IN") AS
    //"TEMP(Calculation_0070818164712315)(1653230849)(0)",   SUM("Rentabilidad_1"."CF") AS
    //"TEMP(Calculation_0070818164712315)(3669921802)(0)",   SUM("Rentabilidad_1"."Rentabilidad") AS
    //"TEMP(Calculation_5560818164729849)(3482281234)(0)",   SUM("Rentabilidad_1"."TOTAL VENTA") AS
    //"TEMP(Calculation_8940818185618064)(293833081)(0)",   SUM("Rentabilidad_1"."TOTAL REPARTO") AS
    //"TEMP(Calculation_8940818185618064)(3492347901)(0)",   SUM("Rentabilidad_1"."Calculation_1880818214638259") AS
    //"sum:Calculation_1880818214638259:ok" FROM "Rentabilidad_1" WHERE (("Rentabilidad_1"."Locación" = \'Bogota Sur\') AND
    //("Rentabilidad_1"."Zona" = \'CE\')) GROUP BY "Rentabilidad_1"."GEC (group)";

    // TRANSLATION

    // SELECT "Rentabilidad_1"."GEC (group)" AS "GEC (group)",   SUM("Rentabilidad_1"."IN") AS
    //"TEMP(Calculation_0070818164712315)(1653230849)(0)",   SUM("Rentabilidad_1"."CF") AS
    //"TEMP(Calculation_0070818164712315)(3669921802)(0)",   SUM("Rentabilidad_1"."Rentabilidad") AS
    //"TEMP(Calculation_5560818164729849)(3482281234)(0)",   SUM("Rentabilidad_1"."TOTAL VENTA") AS
    //"TEMP(Calculation_8940818185618064)(293833081)(0)",   SUM("Rentabilidad_1"."TOTAL REPARTO") AS
    //"TEMP(Calculation_8940818185618064)(3492347901)(0)",   SUM("Rentabilidad_1"."Calculation_1880818214638259") AS
    //"sum:Calculation_1880818214638259:ok" FROM "Rentabilidad_1" WHERE (("Rentabilidad_1"."Locación" = 4) AND
    //("Rentabilidad_1"."Zona" = 0)) GROUP BY "Rentabilidad_1"."GEC (group)";



    // *****************************************************************************************************************
    // * Column generation
    // *****************************************************************************************************************


    auto gec_group = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/140.bin");
    auto in = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/46.bin");
    auto cf = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/13.bin");
    auto rentabilidad = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/99.bin");
    auto total_venta = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/124.bin");
    auto total_reparto = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/122.bin");
    //auto calculation = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/15.bin");
    auto locacion = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/54.bin");
    auto zona = binary_io<uncompr_f>::load("public_bi_data/rentabilidad/139.bin");



    // *****************************************************************************************************************
    // * Query execution
    // *****************************************************************************************************************


    using ps = scalar<v64 < uint64_t>>;

    auto intermediate_1 = select<equals, ps, uncompr_f, uncompr_f>(locacion, 4);
    auto intermediate_2 = select<equals, ps, uncompr_f, uncompr_f>(zona, 0);
    auto intermediate_3 = intersect_sorted<ps, uncompr_f, uncompr_f>(intermediate_1, intermediate_2);

    auto intermediate_4 = project<ps, uncompr_f>(gec_group, intermediate_3);
    auto intermediate_5 = project<ps, uncompr_f>(in, intermediate_3);
    auto intermediate_6 = project<ps, uncompr_f>(cf, intermediate_3);
    auto intermediate_7 = project<ps, uncompr_f>(rentabilidad, intermediate_3);
    auto intermediate_8 = project<ps, uncompr_f>(total_venta, intermediate_3);
    auto intermediate_9 = project<ps, uncompr_f>(total_reparto, intermediate_3);
    //auto intermediate_10 = project<ps, uncompr_f>(calculation, intermediate_3);

    auto grouped = group<ps, uncompr_f, uncompr_f>(intermediate_4);
    auto group_ids = std::get<0>(grouped);
    auto groups = std::get<1>(grouped);

    auto intermediate_11 = agg_sum<ps, uncompr_f>(group_ids, intermediate_5, groups->get_count_values());
    auto intermediate_12 = agg_sum<ps, uncompr_f>(group_ids, intermediate_6, groups->get_count_values());
    auto intermediate_13 = agg_sum<ps, uncompr_f>(group_ids, intermediate_7, groups->get_count_values());
    auto intermediate_14 = agg_sum<ps, uncompr_f>(group_ids, intermediate_8, groups->get_count_values());
    auto intermediate_15 = agg_sum<ps, uncompr_f>(group_ids, intermediate_9, groups->get_count_values());
    //auto intermediate_16 = agg_sum<ps, uncompr_f>(group_ids, intermediate_10, groups->get_count_values());

    auto group_repr = project<ps, uncompr_f>(intermediate_4, groups);





    // *****************************************************************************************************************
    // * Result output
    // *****************************************************************************************************************


    print_columns(print_buffer_base::decimal, group_repr, intermediate_11, intermediate_12, intermediate_13,
                  intermediate_14, intermediate_15,
                  "GEC (group)", "TEMP(Calculation_0070818164712315)(1653230849)(0)",
                  "TEMP(Calculation_0070818164712315)(3669921802)(0)",
                  "TEMP(Calculation_5560818164729849)(3482281234)(0)",
                  "TEMP(Calculation_8940818185618064)(293833081)(0)");


    return 0;
}


