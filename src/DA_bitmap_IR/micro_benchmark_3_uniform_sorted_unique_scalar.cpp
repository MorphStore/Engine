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
 * @file micro_benchmark_3_uniform_sorted_scalar.h
 * @brief Experimental Evaluation:
 *              (3) Intermediate Representation (IR) - Transformations:
 *                  - Base data: uniform + unique + sorted ASC integers (64-bit) with values 0 and TEST_DATA_COUNT-1
 *                  - Transformations scalar:
 *                      (1) PL -> BM
 *                      (2) BM -> PL
 *                  - Measure execution time + memory footprint
 *                  - Results are written to: micro_benchmark_3_asc_scalar.csv
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/morphing/wah.h>
#include <core/utils/printing.h>

#include <core/morphing/intermediates/transformations/transformation_algorithms.h>

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

    // scalar-processing:
    using processingStyle = scalar<v64<uint64_t>>;

    // Generate for each selectivity unique, sorted, uniform distributed base data and execute transformation + measure
    std::vector<double> selectivities = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    // hash map to store results: key = selectivity, value = pair of {exec. time (μs), memory footprint (B)}
    // two measurements: PL -> BM && BM -> PL
    std::unordered_map<double, std::pair<std::chrono::microseconds , size_t>> ir_transformation_pl_to_bm_results;
    std::unordered_map<double, std::pair<std::chrono::microseconds , size_t>> ir_transformation_bm_to_pl_results;

    for(auto selectivity : selectivities) {
        const size_t countPosLog = static_cast<size_t>(
                TEST_DATA_COUNT * selectivity
        );

        // --------------- (1) Generate test data ---------------
        // uniform distributed + unique + sorted ASC (to simulate a position-list as intermediate)
        auto inPosCol = reinterpret_cast< const column< position_list_f<uncompr_f> > * >(
                generate_sorted_unique_extraction(countPosLog, TEST_DATA_COUNT)
        );

        // --------------- (2) IR-Transformation: PL -> BM ---------------

        // clear cache before measurement
        clear_cache();

        auto plToBm_start = high_resolution_clock::now();

        auto plToBmCol =
                transform_IR<
                    processingStyle,
                    bitmap_f<>,
                    position_list_f<>
                >(inPosCol);

        auto plToBm_end = high_resolution_clock::now();
        auto plToBm_exec_time = duration_cast<microseconds>(plToBm_end - plToBm_start);
        auto plToBm_used_bytes = plToBmCol->get_size_used_byte();

        // store results for position-list
        if(ir_transformation_pl_to_bm_results.count(selectivity) == 0){ // store only, if the selectivity does not exist so far...
            ir_transformation_pl_to_bm_results.insert({selectivity, {plToBm_exec_time, plToBm_used_bytes}});
        }

        // --------------- (3) IR-Transformation: BM -> PL ---------------

        // clear cache before measurement
        clear_cache();

        auto bmToPl_start = high_resolution_clock::now();

        auto bmToPlCol =
                transform_IR<
                    processingStyle,
                    position_list_f<>,
                    bitmap_f<>
                >(plToBmCol);

        auto bmToPl_end = high_resolution_clock::now();
        auto bmToPl_exec_time = duration_cast<microseconds>(bmToPl_end - bmToPl_start);
        auto bmToPl_used_bytes = bmToPlCol->get_size_used_byte();

        // store results for position-list
        if(ir_transformation_bm_to_pl_results.count(selectivity) == 0){ // store only, if the selectivity does not exist so far...
            ir_transformation_bm_to_pl_results.insert({selectivity, {bmToPl_exec_time, bmToPl_used_bytes}});
        }
    }

    // --------------- (3) Write results to file ---------------
    // store results of each IR transformation as triple (selectivity, time in ms, memory in B) in csv
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_3_uniform_sorted_unique_scalar.csv");

    mapStream << "\"PlToBm:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\",\"memory (B)\"" << "\n";
    for(auto& element : ir_transformation_pl_to_bm_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second
                  << "\n";
    }
    mapStream << "\"endOfPlToBmResults\"\n";
    mapStream << "\"BmToPl:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\",\"memory (B)\"" << "\n";
    for(auto& element : ir_transformation_bm_to_pl_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second
                  << "\n";
    }
    mapStream << "\"endOfBmToPlResults\"\n";
    mapStream.close();

    return 0;
}