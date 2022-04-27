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
 * @file micro_benchmark_1_asc_scalar.h
 * @brief Experimental Evaluation:
 *              (1) Simple Selection Query:
 *                  - Base data: values between 0 and TEST_DATA_COUNT-1 => sorted + unique => ASC
 *                  - Selectivity variation between 0 and 1 (using 0.1 steps)
 *                  - Compare: Bitmap vs. Position-List
 *                  - Results are written to: micro_benchmark_1_asc_scalar_results.txt
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
#include <map>
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

    // selectivity ranges
    //std::vector<double> selectivity = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector<double> selectivity = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}; // more fine-granular
    // result maps: key = selectivity, value = pair of (execution time in millisec., memory in bytes)
    std::map<double, std::pair<std::chrono::duration<float, std::milli>, size_t>> position_list_results;
    std::map<double, std::pair<std::chrono::duration<float, std::milli>, size_t>> bitmap_results;

    // number elements to flush cache in clear_cache:
    const size_t cacheElements = 10 * 1000 * 1000;

    // --------------- (1) Generate test data ---------------
    auto inCol = generate_sorted_unique( TEST_DATA_COUNT, 0, 1);

    // --------------- (2.1) Select-operator using position-list ---------------
    for(auto s : selectivity){
        const uint64_t predicate = TEST_DATA_COUNT * s;

        // clear cache before measurement
        clear_cache(cacheElements); // around 8 MB of data

        auto start = high_resolution_clock::now();

        auto result_pl =
                morphstore::select<
                    less,
                    scalar<v64<uint64_t>>,
                    position_list_f<uncompr_f>,
                    uncompr_f
                >(inCol, predicate);

        auto end = high_resolution_clock::now();
        auto exec_time = duration_cast<milliseconds>(end - start);
        auto used_bytes = result_pl->get_size_used_byte();

        // store result in map
        position_list_results.insert({s, {exec_time, used_bytes}});
    }

    // --------------- (2.2) Select-operator using bitmap ---------------
    for(auto s : selectivity){
        const uint64_t predicate = TEST_DATA_COUNT * s;

        // clear cache before measurement
        clear_cache(cacheElements); // around 8 MB of data

        auto start = high_resolution_clock::now();

        auto result_bm =
                morphstore::select<
                    less,
                    scalar<v64<uint64_t>>,
                    bitmap_f<uncompr_f>,
                    uncompr_f
                >(inCol, predicate);

        auto end = high_resolution_clock::now();
        auto exec_time = duration_cast<milliseconds>(end - start);
        auto used_bytes = result_bm->get_size_used_byte();

        // store result in map
        bitmap_results.insert({s, {exec_time, used_bytes}});
    }

    // --------------- (3) Write results to file ---------------
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_1_asc_scalar.txt");

    mapStream << "Position-List: " << "\n";
    std::map<double, std::pair<std::chrono::duration<float, std::milli>, size_t>>::iterator it_pl;
    for(it_pl = position_list_results.begin(); it_pl != position_list_results.end(); it_pl++){
        mapStream << "select. = " << (*it_pl).first
            << ": exec.(ms) = " << (*it_pl).second.first.count()
            << ", memory(B) = " << (*it_pl).second.second
            << "\n";
    }
    mapStream << "Bitmap: " << "\n";
    std::map<double, std::pair<std::chrono::duration<float, std::milli>, size_t>>::iterator it_bm;
    for(it_bm = bitmap_results.begin(); it_bm != bitmap_results.end(); it_bm++){
        mapStream << "selectivity = " << (*it_bm).first
                  << ": exec. (ms) = " << (*it_bm).second.first.count()
                  << ", memory (B) = " << (*it_bm).second.second
                  << "\n";
    }

    mapStream.close();

    return 0;
}
