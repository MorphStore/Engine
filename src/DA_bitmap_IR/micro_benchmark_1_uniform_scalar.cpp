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
 * @file micro_benchmark_1_uniform_scalar.h
 * @brief Experimental Evaluation:
 *              (1) Simple Selection Query:
 *                  - Base data: uniform distribution with values 0 and TEST_DATA_COUNT-1
 *                  - Selectivity variation between 0 and 1
 *                  - Compare: Bitmap vs. Position-List
 *                  - Results are written to: micro_benchmark_1_uniform_scalar_results.txt
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/operators/general_vectorized/select_bm_uncompr.h>
#include <core/operators/general_vectorized/select_pl_uncompr.h>

#include <vector>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <fstream>

#define TEST_DATA_COUNT 1000 * 1000

using namespace morphstore;
using namespace vectorlib;
using namespace std::chrono;

// function to ensure that the cache is flushed
void clear_cache(const size_t size) {
    uint64_t *tmp = new uint64_t[size];
    for(size_t i = 0; i < size; i++)
    {
        tmp[i] = rand();
    }
}

int main( void ) {

    // hash map for results: for each intermediate data structure: key = selectivity, value = pair(store execution time in ms, memory footprint in bytes)
    // + no collision handling, just skipping if the selectivity key already exists TODO: add collision-handling?
    std::unordered_map<double, std::pair<std::chrono::duration<float, std::milli>, size_t>> position_list_results;
    std::unordered_map<double, std::pair<std::chrono::duration<float, std::milli>, size_t>> bitmap_results;

    // number elements to flush cache in clear_cache:
    const size_t cacheElements = 10 * 1000 * 1000;

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

    // for each 100th data point in TEST_DATA_COUNT:
    // exec. less-than selection, calculate selectivity + store measurement results for each IR
    for(auto i = 0; i < TEST_DATA_COUNT; i+=100){

        // ********************************* POSITION-LIST *********************************

        // clear cache before measurement
        clear_cache(cacheElements); // around 8 MB of data

        auto pl_start = high_resolution_clock::now();

        // position-list select-operator
        auto pl_result =
                morphstore::select<
                    less,
                    scalar<v64<uint64_t>>,
                    position_list_f<uncompr_f>,
                    uncompr_f
                >(inCol, i);

        auto pl_end = high_resolution_clock::now();
        auto pl_exec_time = duration_cast<milliseconds>(pl_end - pl_start);
        auto pl_used_bytes = pl_result->get_size_used_byte();

        // calculate selectivity
        double selectivity = (double)pl_result->get_count_values() / (double)TEST_DATA_COUNT;

        // store results for position-list
        if(position_list_results.count(selectivity) == 0){ // store only, if the selectivity does not exist so far...
            position_list_results.insert({selectivity, {pl_exec_time, pl_used_bytes}});
        }

        // ********************************* BITMAP *********************************

        // clear cache before measurement
        clear_cache(cacheElements); // around 8 MB of data

        auto bm_start = high_resolution_clock::now();

        // position-list select-operator
        auto bm_result =
                morphstore::select<
                    less,
                    scalar<v64<uint64_t>>,
                    bitmap_f<uncompr_f>,
                    uncompr_f
                >(inCol, i);

        auto bm_end = high_resolution_clock::now();
        auto bm_exec_time = duration_cast<milliseconds>(bm_end - bm_start);
        auto bm_used_bytes = bm_result->get_size_used_byte();

        // store results for position-list
        if(bitmap_results.count(selectivity) == 0) { // store only, if the selectivity does not exist so far...
            bitmap_results.insert({selectivity, {bm_exec_time, bm_used_bytes}});
        }
    }

    // --------------- (3) Write results to file ---------------
    // store results of each IR as triple (selectivity, time in ms, memory in B) in csv
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_1_uniform_scalar.csv");

    mapStream << "PL: " << "\n";
    mapStream << "\"selectivity\",\"execution time (ms)\",\"memory (B)\"" << "\n";
    for(auto& element : position_list_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second
                  << "\n";
    }
    mapStream << "\"endOfPositionListResults\"\n";
    mapStream << "BM: " << "\n";
    mapStream << "\"selectivity\",\"execution time (ms)\",\"memory (B)\"" << "\n";
    for(auto& element : bitmap_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second
                  << "\n";
    }
    mapStream << "endOfBitmapResults\n";
    mapStream.close();

    return 0;
}
