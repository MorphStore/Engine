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
 * @file micro_benchmark_1_uniform_avx512.cpp
 * @brief Experimental Evaluation:
 *              (1) Simple Selection Query:
 *                  - Base data: uniform distribution with values 0 and TEST_DATA_COUNT-1
 *                  - Selectivity variation between 0 and 1
 *                  - Compare: Bitmap vs. Position-List
 *                  - Results are written to: micro_benchmark_1_uniform_avx512.csv
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/intermediates/position_list.h>

#include <vector/simd/avx512/extension_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>
#include <vector/simd/avx512/primitives/compare_avx512.h>

#include <core/operators/general_vectorized/select_bm_uncompr.h>
#include <core/operators/general_vectorized/select_pl_uncompr.h>

#include <vector>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <cmath>

// server:
#define TEST_DATA_COUNT 100 * 1000 * 1000

using namespace morphstore;
using namespace vectorlib;
using namespace std::chrono;

// function to ensure that the cache is flushed
void clear_cache() {
    size_t elements = TEST_DATA_COUNT;
    std::vector<uint64_t> clear = std::vector<uint64_t>();
    clear.resize(elements, 42);
    for (size_t i = 0; i < clear.size(); i++) {
        clear[i] += 1;
    }
    clear.resize(0);
}

int main( void ) {

    // avx512-processing:
    using processingStyle = avx512<v512<uint64_t>>;

    // hash map for results: for each intermediate data structure: key = selectivity, value = pair(store execution time in ms, memory footprint in bytes)
    // + no collision handling, just skipping if the selectivity key already exists TODO: add collision-handling?
    std::unordered_map<double, std::pair<std::chrono::microseconds , size_t>> position_list_results;
    std::unordered_map<double, std::pair<std::chrono::microseconds , size_t>> bitmap_results;

    // --------------- (1) Generate test data ---------------
    auto inCol = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    TEST_DATA_COUNT - 1
            ),
            false
    );

    // --------------- (2) Selection operation ---------------

    // for each i-th data point in TEST_DATA_COUNT:
    // exec. less-than selection, calculate selectivity + store measurement results for each IR
    size_t steps = 1000000;
    for(auto i = 0; i < TEST_DATA_COUNT+1; i += steps){

        // ********************************* POSITION-LIST *********************************

        // clear cache before measurement
        clear_cache();

        auto pl_start = high_resolution_clock::now();

        // position-list select-operator
        auto pl_result =
                morphstore::select<
                    less,
                    processingStyle,
                    position_list_f<uncompr_f>,
                    uncompr_f
                >(inCol, i);

        auto pl_end = high_resolution_clock::now();
        auto pl_exec_time = duration_cast<microseconds>(pl_end - pl_start);
        auto pl_used_bytes = pl_result->get_size_used_byte();

        // calculate selectivity: round up to 2 decimal places (0.XX)
        double selectivity = std::ceil(
                (static_cast<double>(pl_result->get_count_values()) / static_cast<double>(TEST_DATA_COUNT))
                * 100.0) / 100.0;

        // store results for position-list
        if(position_list_results.count(selectivity) == 0){ // store only, if the selectivity does not exist so far...
            position_list_results.insert({selectivity, {pl_exec_time, pl_used_bytes}});
        }

        // ********************************* BITMAP *********************************

        // clear cache before measurement
        clear_cache();

        auto bm_start = high_resolution_clock::now();

        // bitmap select-operator
        auto bm_result =
                morphstore::select<
                    less,
                    processingStyle,
                    bitmap_f<uncompr_f>,
                    uncompr_f
                >(inCol, i);

        auto bm_end = high_resolution_clock::now();
        auto bm_exec_time = duration_cast<microseconds>(bm_end - bm_start);
        auto bm_used_bytes = bm_result->get_size_used_byte();

        // store results for bitmap
        if(bitmap_results.count(selectivity) == 0) { // store only, if the selectivity does not exist so far...
            bitmap_results.insert({selectivity, {bm_exec_time, bm_used_bytes}});
        }
    }

    // --------------- (3) Write results to file ---------------
    // store results of each IR as triple (selectivity, time in ms, memory in B) in csv
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_1_uniform_avx512.csv");

    mapStream << "\"PL:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\",\"memory (B)\"" << "\n";
    for(auto& element : position_list_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second
                  << "\n";
    }
    mapStream << "\"endOfPlResults\"\n";

    mapStream << "\"BM:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\",\"memory (B)\"" << "\n";
    for(auto& element : bitmap_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second
                  << "\n";
    }
    mapStream << "\"endOfBmResults\"\n";
    mapStream.close();

    return 0;
}
