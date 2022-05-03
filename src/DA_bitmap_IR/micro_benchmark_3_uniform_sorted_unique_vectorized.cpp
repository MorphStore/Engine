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
 * @file micro_benchmark_3_uniform_sorted_vectorized.cpp
 * @brief Experimental Evaluation:
 *              (3) Intermediate Representation (IR) - Transformation: only BM -> PL vectorized
 *                  - Base data: uniform + unique + sorted ASC integers (64-bit) with values 0 and TEST_DATA_COUNT-1
 *                  - Compare only BM->PL transformation with scalar, avx2, avx512
 *                  - Measure execution time
 *                  - Results are written to: micro_benchmark_3_uniform_sorted_unique_vectorized.csv
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/morphing/intermediates/bitmap.h>
#include <core/morphing/intermediates/position_list.h>

#include <core/morphing/intermediates/transformations/transformation_algorithms.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <vector/simd/avx512/extension_avx512.h>
#include <vector/simd/avx512/primitives/calc_avx512.h>
#include <vector/simd/avx512/primitives/io_avx512.h>
#include <vector/simd/avx512/primitives/create_avx512.h>
#include <vector/simd/avx512/primitives/compare_avx512.h>

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

    // processing styles:
    using scalar = scalar<v64<uint64_t>>;
    using avx2 = avx2<v256<uint64_t>>;
    using avx512 = avx512<v512<uint64_t>>;

    // Generate for each selectivity unique, sorted, uniform distributed base data and execute transformation + measure
    std::vector<double> selectivities = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};

    // hash map to store results: key = selectivity, value = execution time for each processing style
    std::unordered_map<double, std::chrono::microseconds> bm_to_pl_scalar;
    std::unordered_map<double, std::chrono::microseconds> bm_to_pl_avx2;
    std::unordered_map<double, std::chrono::microseconds> bm_to_pl_avx512;

    for(auto selectivity : selectivities) {

        // --------------- (1) Generate test data ---------------
        // Note: we do not yet have a function to generate bitmap data, so we fist generate the position-list data,
        //       transform it to a bitmap and then execute the transformations according to the processing style
        const size_t countPosLog = static_cast<size_t>(
                TEST_DATA_COUNT * selectivity
        );
        // uniform distributed + unique + sorted ASC (to simulate a position-list as intermediate)
        auto inPosCol = reinterpret_cast< const column< position_list_f<uncompr_f> > * >(
                generate_sorted_unique_extraction(countPosLog, TEST_DATA_COUNT)
        );

        // transform position-list column to bitmap column
        auto bmCol =
                transform_IR<
                    scalar, // only scalar available
                    bitmap_f<>,
                    position_list_f<>
                >(inPosCol);

        // --------------- (2) IR-Transformation: BM -> PL (scalar) ---------------

        // clear cache before measurement
        clear_cache();

        auto bmToPl_scalar_start = high_resolution_clock::now();

        auto bmToPl_scalar =
                transform_IR<
                    scalar,
                    position_list_f<>,
                    bitmap_f<>
                >(bmCol);

        auto bmToPl_scalar_end = high_resolution_clock::now();
        auto bmToPl_scalar_exec_time = duration_cast<microseconds>(bmToPl_scalar_end - bmToPl_scalar_start);

        // store results for position-list
        bm_to_pl_scalar.insert({selectivity, bmToPl_scalar_exec_time});

        // satisfy compiler error 'unused variable'
        (void)bmToPl_scalar;

        // --------------- (2) IR-Transformation: BM -> PL (avx2) ---------------

        // clear cache before measurement
        clear_cache();

        auto bmToPl_avx2_start = high_resolution_clock::now();

        auto bmToPl_avx2 =
                transform_IR<
                    avx2,
                    position_list_f<>,
                    bitmap_f<>
                >(bmCol);

        auto bmToPl_avx2_end = high_resolution_clock::now();
        auto bmToPl_avx2_exec_time = duration_cast<microseconds>(bmToPl_avx2_end - bmToPl_avx2_start);

        // store results for position-list
        bm_to_pl_avx2.insert({selectivity, bmToPl_avx2_exec_time});

        // satisfy compiler error 'unused variable'
        (void)bmToPl_avx2;

        // --------------- (2) IR-Transformation: BM -> PL (avx512) ---------------

        // clear cache before measurement
        clear_cache();

        auto bmToPl_avx512_start = high_resolution_clock::now();

        auto bmToPl_avx512 =
                transform_IR<
                    avx512,
                    position_list_f<>,
                    bitmap_f<>
                >(bmCol);

        auto bmToPl_avx512_end = high_resolution_clock::now();
        auto bmToPl_avx512_exec_time = duration_cast<microseconds>(bmToPl_avx512_end - bmToPl_avx512_start);

        // store results for position-list
        bm_to_pl_avx512.insert({selectivity, bmToPl_avx512_exec_time});

        // satisfy compiler error 'unused variable'
        (void)bmToPl_avx512;
    }

    // --------------- (3) Write results to file ---------------
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_3_uniform_sorted_unique_vectorized.csv");

    // scalar
    mapStream << "\"BM2PL scalar:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : bm_to_pl_scalar){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfBM2PLResults\"\n";

    // avx2
    mapStream << "\"BM2PL avx2:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : bm_to_pl_avx2){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfBM2PLResults\"\n";

    // avx512
    mapStream << "\"BM2PL avx512:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : bm_to_pl_avx512){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfBM2PLResults\"\n";
    mapStream.close();

    return 0;
}