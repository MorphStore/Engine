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
#include "../../include/core/operators/scalar/merge_uncompr.h"
//#include "../../include/core/operators/scalar/calc_uncompr.h"
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
struct equals {
    constexpr bool operator()(const T &lhs, const T &rhs) const
    {
        return lhs == rhs;
    }
};

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


    std::cout << "Eixo_1" << std::endl;

    // SELECT "Eixo_1"."ch_catalogo_guia" AS "ch_catalogo_guia",
    //"Eixo_1"."cod_da_unidade_de_ensino" AS "cod_da_unidade_de_ensino",
    //"Eixo_1"."cod_ibge_do_local_da_oferta" AS "cod_ibge_do_local_da_oferta",
    //"Eixo_1"."codigo_do_curso" AS "codigo_do_curso",
    //COUNT("Eixo_1"."Calculation_838513981462429699") AS "ctd:Calculation_838513981462429699:ok",
    //"Eixo_1"."ead" AS "ead",
    //"Eixo_1"."eixo_tecnologico_catalogo_guia" AS "eixo_tecnologico_catalogo_guia",
    //"Eixo_1"."forma_ingresso" AS "forma_ingresso",
    //"Eixo_1"."municipio_do_local_da_oferta" AS "municipio_do_local_da_oferta",
    //"Eixo_1"."no_dependencia_admin" AS "no_dependencia_admin",
    //"Eixo_1"."no_modalidade" AS "no_modalidade",
    //"Eixo_1"."no_parceiro_demandante" AS "no_parceiro_demandante",
    // "Eixo_1"."no_sistema_ensino" AS "no_sistema_ensino",
    // "Eixo_1"."nome_curso_catalogo_guia" AS "nome_curso_catalogo_guia",
    // "Eixo_1"."nome_da_ue" AS "nome_da_ue",
    // "Eixo_1"."nome_da_uer" AS "nome_da_uer",
    // "Eixo_1"."subtipo_curso" AS "subtipo_curso",
    // "Eixo_1"."uf_do_local_da_oferta" AS "uf_do_local_da_oferta",
    // "Eixo_1"."unidade_demandante" AS "unidade_demandante",
    // "Eixo_1"."data_de_inicio(year)" AS "yr:data_de_inicio:ok"
    // FROM "Eixo_1"
    // WHERE (
    // ("Eixo_1"."data_de_inicio(year)" = 2013
    // OR "Eixo_1"."data_de_inicio(year)" = 2014
    // OR "Eixo_1"."data_de_inicio(year)" = 2015)
    // AND (
    // ("Eixo_1"."nome da sit matricula (situacao detalhada)" != 2
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 3
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 4
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 5
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 9
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 12
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 14
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 15
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 18
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 17
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 23
    // AND "Eixo_1"."nome da sit matricula (situacao detalhada)" != 24) AND
    // ("Eixo_1"."nome da sit matricula (situacao detalhada)" != 18446744073709551615))
    // AND (
    // "Eixo_1"."situacao_da_turma" != 0
    // AND "Eixo_1"."situacao_da_turma" != 3
    // AND "Eixo_1"."situacao_da_turma" != 5)
    // )
    // GROUP BY "ch_catalogo_guia",
    // "cod_da_unidade_de_ensino",
    // "cod_ibge_do_local_da_oferta",
    // "Eixo_1"."codigo_do_curso",
    // "ead",
    // "eixo_tecnologico_catalogo_guia",
    // "forma_ingresso",
    // "municipio_do_local_da_oferta",
    // "no_dependencia_admin",
    // "no_modalidade",
    // "no_parceiro_demandante",
    // "no_sistema_ensino",
    // "nome_curso_catalogo_guia",
    // "nome_da_ue",
    // "nome_da_uer",
    // "subtipo_curso",
    // "uf_do_local_da_oferta",
    // "unidade_demandante",
    // "yr:data_de_inicio:ok",
    // "codigo_do_curso";
    


    // *****************************************************************************************************************
    // * Column generation
    // *****************************************************************************************************************


    auto ch_catalogo_guia = binary_io<uncompr_f>::load("public_bi_data/eixo/11.bin");
    auto cod_da_unidade_de_ensino = binary_io<uncompr_f>::load("public_bi_data/eixo/18.bin");
    auto cod_ibge_do_local_da_oferta = binary_io<uncompr_f>::load("public_bi_data/eixo/20.bin");
    auto codigo_do_curso = binary_io<uncompr_f>::load("public_bi_data/eixo/22.bin");
    //auto calculation_838513981462429699 = binary_io<uncompr_f>::load("public_bi_data/eixo/1.bin");
    auto ead = binary_io<uncompr_f>::load("public_bi_data/eixo/46.bin");
    auto eixo_tecnologico_catalogo_guia = binary_io<uncompr_f>::load("public_bi_data/eixo/49.bin");
    auto forma_ingresso = binary_io<uncompr_f>::load("public_bi_data/eixo/53.bin");
    auto municipio_do_local_da_oferta = binary_io<uncompr_f>::load("public_bi_data/eixo/57.bin");
    auto no_dependencia_admin = binary_io<uncompr_f>::load("public_bi_data/eixo/58.bin");
    auto no_modalidade = binary_io<uncompr_f>::load("public_bi_data/eixo/59.bin");
    auto no_parceiro_demandante = binary_io<uncompr_f>::load("public_bi_data/eixo/60.bin");
    auto no_sistema_ensino = binary_io<uncompr_f>::load("public_bi_data/eixo/62.bin");
    auto nome_curso_catalogo_guia = binary_io<uncompr_f>::load("public_bi_data/eixo/66.bin");
    auto nome_da_ue = binary_io<uncompr_f>::load("public_bi_data/eixo/67.bin");
    auto nome_da_uer = binary_io<uncompr_f>::load("public_bi_data/eixo/68.bin");
    auto subtipo_curso = binary_io<uncompr_f>::load("public_bi_data/eixo/82.bin");
    auto uf_do_local_da_oferta = binary_io<uncompr_f>::load("public_bi_data/eixo/90.bin");
    auto unidade_demandante = binary_io<uncompr_f>::load("public_bi_data/eixo/91.bin");
    auto data_de_inicio_year = binary_io<uncompr_f>::load("public_bi_data/eixo/35.bin");
    auto nome_da_sit_matricula_situacao_detalhada = binary_io<uncompr_f>::load("public_bi_data/eixo/65.bin");
    auto situacao_da_turma = binary_io<uncompr_f>::load("public_bi_data/eixo/78.bin");

    auto count_column = binary_io<uncompr_f>::load("public_bi_data/eixo/const_4.bin");

    // *****************************************************************************************************************
    // * Query execution
    // *****************************************************************************************************************


    using ps = scalar<v64<uint64_t>>;

    // WHERE clause

    auto select_1 = select<equals, ps, uncompr_f, uncompr_f>(data_de_inicio_year, 2013);
    auto select_2 = select<equals, ps, uncompr_f, uncompr_f>(data_de_inicio_year, 2014);
    auto select_3 = select<equals, ps, uncompr_f, uncompr_f>(data_de_inicio_year, 2015);

    auto or_1 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f>(select_1, select_2);
    auto or_2 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f>(or_1, select_3);


    auto select_4 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 2);
    auto select_5 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 3);
    auto select_6 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 4);
    auto select_7 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 5);
    auto select_8 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 9);
    auto select_9 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 12);
    auto select_10 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 14);
    auto select_11 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 15);
    auto select_12 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 18);
    auto select_13 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 17);
    auto select_14 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 23);
    auto select_15 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 24);

    auto and_1 = intersect_sorted<ps, uncompr_f, uncompr_f>(select_4, select_5);
    auto and_2 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_1, select_6);
    auto and_3 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_2, select_7);
    auto and_4 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_3, select_8);
    auto and_5 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_4, select_9);
    auto and_6 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_5, select_10);
    auto and_7 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_6, select_11);
    auto and_8 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_7, select_12);
    auto and_9 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_8, select_13);
    auto and_10 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_9, select_14);
    auto and_11 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_10, select_15);

    auto select_16 = select<not_equals, ps, uncompr_f, uncompr_f>(nome_da_sit_matricula_situacao_detalhada, 18446744073709551615U);

    auto and_12 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_11, select_16);

    auto and_13 = intersect_sorted<ps, uncompr_f, uncompr_f>(or_2, and_12);

    auto select_17 = select<not_equals, ps, uncompr_f, uncompr_f>(situacao_da_turma, 0);
    auto select_18 = select<not_equals, ps, uncompr_f, uncompr_f>(situacao_da_turma, 3);
    auto select_19 = select<not_equals, ps, uncompr_f, uncompr_f>(situacao_da_turma, 5);

    auto and_14 = intersect_sorted<ps, uncompr_f, uncompr_f>(select_17, select_18);
    auto and_15 = intersect_sorted<ps, uncompr_f, uncompr_f>(and_14, select_19);

    auto where_clause = intersect_sorted<ps, uncompr_f, uncompr_f>(and_13, and_15);

    auto ch_catalogo_guia_proj = project<ps, uncompr_f>(ch_catalogo_guia, where_clause);
    auto cod_da_unidade_de_ensino_proj = project<ps, uncompr_f>(cod_da_unidade_de_ensino, where_clause);
    auto cod_ibge_do_local_da_oferta_proj = project<ps, uncompr_f>(cod_ibge_do_local_da_oferta, where_clause);
    auto codigo_do_curso_proj = project<ps, uncompr_f>(codigo_do_curso, where_clause);
    auto ead_proj = project<ps, uncompr_f>(ead, where_clause);
    auto eixo_tecnologico_catalogo_guia_proj = project<ps, uncompr_f>(eixo_tecnologico_catalogo_guia, where_clause);
    auto forma_ingresso_proj = project<ps, uncompr_f>(forma_ingresso, where_clause);
    auto municipio_do_local_da_oferta_proj = project<ps, uncompr_f>(municipio_do_local_da_oferta, where_clause);
    auto no_dependencia_admin_proj = project<ps, uncompr_f>(no_dependencia_admin, where_clause);
    auto no_modalidade_proj = project<ps, uncompr_f>(no_modalidade, where_clause);
    auto no_parceiro_demandante_proj = project<ps, uncompr_f>(no_parceiro_demandante, where_clause);
    auto no_sistema_ensino_proj = project<ps, uncompr_f>(no_sistema_ensino, where_clause);
    auto nome_curso_catalogo_guia_proj = project<ps, uncompr_f>(nome_curso_catalogo_guia, where_clause);
    auto nome_da_ue_proj = project<ps, uncompr_f>(nome_da_ue, where_clause);
    auto nome_da_uer_proj = project<ps, uncompr_f>(nome_da_uer, where_clause);
    auto subtipo_curso_proj = project<ps, uncompr_f>(subtipo_curso, where_clause);
    auto uf_do_local_da_oferta_proj = project<ps, uncompr_f>(uf_do_local_da_oferta, where_clause);
    auto unidade_demandante_proj = project<ps, uncompr_f>(unidade_demandante, where_clause);
    //auto calculation_838513981462429699_proj = project<ps, uncompr_f>(calculation_838513981462429699, where_clause);
    auto data_de_inicio_year_proj = project<ps, uncompr_f>(data_de_inicio_year, where_clause);
    auto count_column_proj = project<ps, uncompr_f>(count_column, where_clause);

    auto group_1 = group<ps, uncompr_f, uncompr_f>(ch_catalogo_guia_proj);
    auto group_2 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_1), cod_da_unidade_de_ensino_proj);
    auto group_3 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_2), cod_ibge_do_local_da_oferta_proj);
    auto group_4 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_3), codigo_do_curso_proj);
    auto group_5 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_4), ead_proj);
    auto group_6 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_5), eixo_tecnologico_catalogo_guia_proj);
    auto group_7 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_6), forma_ingresso_proj);
    auto group_8 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_7), municipio_do_local_da_oferta_proj);
    auto group_9 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_8), no_dependencia_admin_proj);
    auto group_10 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_9), no_modalidade_proj);
    auto group_11 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_10), no_parceiro_demandante_proj);
    auto group_12 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_11), no_sistema_ensino_proj);
    auto group_13 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_12), nome_curso_catalogo_guia_proj);
    auto group_14 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_13), nome_da_ue_proj);
    auto group_15 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_14), nome_da_uer_proj);
    auto group_16 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_15), subtipo_curso_proj);
    auto group_17 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_16), uf_do_local_da_oferta_proj);
    auto group_18 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_17), unidade_demandante_proj);
    auto group_19 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_18), data_de_inicio_year_proj);
    auto group_20 = group<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(std::get<0>(group_19), codigo_do_curso_proj);

    auto group_ids = std::get<0>(group_20);
    auto groups = std::get<1>(group_20);

    auto counts = agg_sum<ps, uncompr_f>(group_ids, count_column_proj, groups->get_count_values());

    auto output_1 = project<ps, uncompr_f>(ch_catalogo_guia_proj, groups);
    auto output_2 = project<ps, uncompr_f>(cod_da_unidade_de_ensino_proj, groups);
    auto output_3 = project<ps, uncompr_f>(cod_ibge_do_local_da_oferta_proj, groups);
    auto output_4 = project<ps, uncompr_f>(codigo_do_curso_proj, groups);
    auto output_5 = counts;
    auto output_6 = project<ps, uncompr_f>(ead_proj, groups);
    /*auto output_7 = project<ps, uncompr_f>(eixo_tecnologico_catalogo_guia_proj, groups);
    auto output_8 = project<ps, uncompr_f>(forma_ingresso_proj, groups);
    auto output_9 = project<ps, uncompr_f>(municipio_do_local_da_oferta_proj, groups);
    auto output_10 = project<ps, uncompr_f>(no_dependencia_admin_proj, groups);
    auto output_11 = project<ps, uncompr_f>(no_modalidade_proj, groups);
    auto output_12 = project<ps, uncompr_f>(no_parceiro_demandante_proj, groups);
    auto output_13 = project<ps, uncompr_f>(no_sistema_ensino_proj, groups);
    auto output_14 = project<ps, uncompr_f>(nome_curso_catalogo_guia_proj, groups);
    auto output_15 = project<ps, uncompr_f>(nome_da_ue_proj, groups);
    auto output_16 = project<ps, uncompr_f>(nome_da_uer_proj, groups);
    auto output_17 = project<ps, uncompr_f>(subtipo_curso_proj, groups);
    auto output_18 = project<ps, uncompr_f>(uf_do_local_da_oferta_proj, groups);
    auto output_19 = project<ps, uncompr_f>(unidade_demandante_proj, groups);
    auto output_20 = project<ps, uncompr_f>(data_de_inicio_year_proj, groups);*/



    // *****************************************************************************************************************
    // * Result output
    // *****************************************************************************************************************


    /*print_columns(print_buffer_base::decimal, output_1, output_2, output_3, output_4, output_5, output_6, output_7,
            output_8, output_9, output_10, output_11, output_12, output_13, output_14, output_15, output_16, output_17,
            output_18, output_19, output_20,
            "ch_catalogo_guia", "cod_da_unidade_de_ensino", "cod_ibge_do_local_da_oferta", "codigo_do_curso",
            "ctd:Calculation_838513981462429699:ok", "ead", "eixo_tecnologico_catalogo_guia", "forma_ingresso",
            "municipio_do_local_da_oferta", "no_dependencia_admin", "no_modalidade", "no_parceiro_demandante",
            "no_sistema_ensino", "nome_curso_catalogo_guia", "nome_da_ue", "nome_da_uer", "subtipo_curso",
            "uf_do_local_da_oferta", "unidade_demandante", "yr:data_de_inicio:ok");*/

    print_columns(print_buffer_base::decimal, output_1, output_2, output_3, output_4, output_5, output_6,
                  "ch_catalogo_guia", "cod_da_unidade_de_ensino", "cod_ibge_do_local_da_oferta", "codigo_do_curso",
                  "ctd:Calculation_838513981462429699:ok", "ead");



    return 0;
}

