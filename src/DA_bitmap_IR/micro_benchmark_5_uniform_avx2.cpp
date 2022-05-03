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
 * @file micro_benchmark_5_uniform_avx2.cpp
 * @brief Experimental Evaluation:
 *              (5) Unified Processing Approach: Simple query
 *                  - Base data: uniform distribution with values 0 and TEST_DATA_COUNT-1
 *                  - Query: SELECT SUM(baseCol2) WHERE baseCol1 = 150
 *                  - Measure execution time with
 *                      (1) Query with selection using only position-list processing
 *                      (2) Query with selection using bitmap processing + position-list as output] => internal transformation
 *                  - Results are written to: micro_benchmark_5_uniform_avx2.csv
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/intermediates/position_list.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/operators/general_vectorized/project_compr.h>
#include <core/operators/general_vectorized/select_pl_compr.h>
#include <core/operators/general_vectorized/select_bm_compr.h>

#include <vector>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <cmath>

// local:
//#define TEST_DATA_COUNT 1000

// server:
#define TEST_DATA_COUNT  100 * 1000 * 1000

using namespace morphstore;
using namespace vectorlib;
using namespace std::chrono;

// function to ensure that the cache is flushed
void clear_cache() {
    // local cache: 3072 KB
    //size_t elements = 400 * 1000;
    // server cache: 1024 KB
    size_t elements = 10 * 1000 * 1000;
    std::vector<uint64_t> clear = std::vector<uint64_t>();
    clear.resize(elements, 42);
    for (size_t i = 0; i < clear.size(); i++) {
        clear[i] += 1;
    }
    clear.resize(0);
}

int main( void ) {
// ****************************************************************************
// * Query: SELECT SUM(baseCol2) WHERE baseCol1 < 150
// ****************************************************************************

    // vectorized processing style
    using processingStyle = avx2<v256<uint64_t>>;

    // --------------- (1) Generate test data ---------------

    auto baseCol1 = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    TEST_DATA_COUNT - 1
            ),
            false
    );

    auto baseCol2 = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(0, 10),
            false
    );

    // hash map for results: key = selectivity, value = query execution time
    std::unordered_map<double, std::chrono::microseconds> query_pl_no_transformation_results;
    std::unordered_map<double, std::chrono::microseconds> query_bm_transformation_results; // includes transformation

    // for each i-th data point in TEST_DATA_COUNT: execute query, calculate selectivity + store measurement results for each IR
    //size_t steps = 100;
    // server:
    size_t steps = 1000000;
    for(auto i = 0; i < TEST_DATA_COUNT+1; i += steps){

        // ******************* (2) Query execution using POSITION-LIST select-operator (no transformation) *******************
        clear_cache();

        auto pl_start = high_resolution_clock::now();

        // Positions fulfilling "baseCol1 < 150"
        auto i1_pl =
                select_pl_wit_t<
                    less,
                    processingStyle,
                    position_list_f<uncompr_f>,
                    uncompr_f
                >::apply(baseCol1, i);

        // Data elements of "baseCol2" fulfilling "baseCol1 < 150"
        auto i1_pl_cast = reinterpret_cast< const column< uncompr_f > * >(i1_pl);
        auto i2_pl =
                my_project_wit_t<
                    processingStyle,
                    uncompr_f,
                    uncompr_f,
                    uncompr_f
                >::apply(baseCol2, i1_pl_cast);

        // Sum over the data elements of "baseCol2" fulfilling "baseCol1 < 150"
        auto i3_pl = agg_sum<processingStyle, uncompr_f>(i2_pl);

        auto pl_end = high_resolution_clock::now();
        auto query_pl_exec_time = duration_cast<microseconds>(pl_end - pl_start);

        // calculate selectivity within selection (i1): round up to 2 decimal places (0.XX)
        double selectivity = std::ceil(
                (static_cast<double>(i1_pl->get_count_values()) / static_cast<double>(TEST_DATA_COUNT))
                * 100.0) / 100.0;

        if(query_pl_no_transformation_results.count(selectivity) == 0){ // store only, if the selectivity does not exist so far...
            query_pl_no_transformation_results.insert({selectivity, query_pl_exec_time});
        }

        // satisfy compiler error:
        (void)i3_pl;

        // ******************* (3) Query execution using BITMAP select-operator with position-list output (IR-transformation) *******************

        clear_cache();

        auto bm_start = high_resolution_clock::now();

        // Positions fulfilling "baseCol1 < 150"
        auto i1_bm =
                     select_bm_wit_t<
                     less,
                processingStyle,
                position_list_f<uncompr_f>, // internal IR-transformation to position-list
                uncompr_f
                >::apply(baseCol1, i);

        // Data elements of "baseCol2" fulfilling "baseCol1 < 150"
        auto i1_bm_cast = reinterpret_cast< const column< uncompr_f > * >(i1_bm);
        auto i2_bm =
                my_project_wit_t<
                        processingStyle,
                        uncompr_f,
                        uncompr_f,
                        uncompr_f
                >::apply(baseCol2, i1_bm_cast);

        // Sum over the data elements of "baseCol2" fulfilling "baseCol1 < 150"
        auto i3_bm = agg_sum<processingStyle, uncompr_f>(i2_bm);

        auto bm_end = high_resolution_clock::now();
        auto query_bm_exec_time = duration_cast<microseconds>(bm_end - bm_start);

        if(query_bm_transformation_results.count(selectivity) == 0){
            query_bm_transformation_results.insert({selectivity, query_bm_exec_time});
        }

        // satisfy compiler error:
        (void)i3_bm;
    }

    // --------------- (3) Write results to file ---------------
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_5_uniform_avx2.csv");

    mapStream << "\"Query_Without_Transformation:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : query_pl_no_transformation_results){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfPLResults\"\n";
    mapStream << "\"Query_With_Transformation:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : query_bm_transformation_results){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfBMResults\"\n";
    mapStream.close();

    return 0;
}

