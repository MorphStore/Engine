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
 * @file micro_benchmark_4_uniform_sorted_unique_avx512.cpp
 * @brief Experimental Evaluation:
 *              (4) Intersection / Union between two intermediate representations:
 *                  - Base data: uniform + unique + sorted ASC integers (64-bit) with values 0 and TEST_DATA_COUNT-1
 *                  - bitmap vs position-list, avx512 execution
 *                  - Measure execution time
 *                  - Results are written to: micro_benchmark_4_uniform_sorted_unique_avx512.csv
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
#include <vector/simd/avx512/primitives/manipulate_avx512.h>
#include <vector/simd/avx512/primitives/logic_avx512.h>
#include <vector/simd/avx512/primitives/extract_avx512.h>

#include <core/operators/general_vectorized/intersect_uncompr.h>
#include <core/operators/general_vectorized/intersect_bm_uncompr.h>
#include <core/operators/general_vectorized/merge_uncompr.h>
#include <core/operators/general_vectorized/merge_bm_uncompr.h>

#include <core/morphing/intermediates/transformations/transformation_algorithms.h>

#include <vector>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <cmath>

// server:
#define TEST_DATA_COUNT  100 * 1000 * 1000

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

    // vectorized-processing:
    using processingStyle = avx512<v512<uint64_t>>;

    // Generate for each selectivity unique, sorted, uniform distributed base data and execute union / intersection + measure
    std::vector<double> selectivities = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};

    // hash map to store results: key = selectivity, value = execution time
    std::unordered_map<double, std::chrono::microseconds> merge_bm_avx512;
    std::unordered_map<double, std::chrono::microseconds> merge_pl_avx512;

    std::unordered_map<double, std::chrono::microseconds> intersect_bm_avx512;
    std::unordered_map<double, std::chrono::microseconds> intersect_pl_avx512;

    for(auto selectivity : selectivities) {

        // ********************************* POSITION-LIST ******************************************************************
        // ---------------  Generate test data ---------------
        const size_t countPosLog = static_cast<size_t>(
                TEST_DATA_COUNT * selectivity
        );

        // uniform distributed + unique + sorted ASC
        auto plCol_1 = generate_sorted_unique_extraction(countPosLog, TEST_DATA_COUNT);
        auto plCol_2 = generate_sorted_unique_extraction(countPosLog, TEST_DATA_COUNT);

        // reset cache before measurement
        clear_cache();

        auto intersect_pl_start = high_resolution_clock::now();

        // intersection:
        auto intersect_pl =
                intersect_sorted<
                    processingStyle,
                    uncompr_f,
                    uncompr_f,
                    uncompr_f
                >( plCol_1, plCol_2 );

        auto intersect_pl_end = high_resolution_clock::now();
        auto intersect_pl_exec_time = duration_cast<microseconds>(intersect_pl_end - intersect_pl_start);

        // store result:
        intersect_pl_avx512.insert({selectivity, intersect_pl_exec_time});
        (void)intersect_pl; // satisfy compiler error

        // reset cache before measurement
        clear_cache();

        auto union_pl_start = high_resolution_clock::now();

        // union:
        auto union_pl =
                merge_sorted<
                    processingStyle,
                    uncompr_f,
                    uncompr_f,
                    uncompr_f
                >( plCol_1, plCol_2 );

        auto union_pl_end = high_resolution_clock::now();
        auto union_pl_exec_time = duration_cast<microseconds>(union_pl_end - union_pl_start);

        // store result:
        merge_pl_avx512.insert({selectivity, union_pl_exec_time});
        (void)union_pl; // satisfy compiler error

        // ********************************* BITMAP ******************************************************************

        // transform input-position lists to bitmaps to keep the same data for measurements (no other workaround, expensive)
        auto plCol_1_cast = reinterpret_cast<const column< position_list_f<uncompr_f> > *>(plCol_1);
        auto bmCol_1 =
                transform_IR<
                    processingStyle,
                    bitmap_f<>,
                    position_list_f<>
                >(plCol_1_cast);

        auto plCol_2_cast = reinterpret_cast<const column< position_list_f<uncompr_f> > *>(plCol_2);
        auto bmCol_2 =
                transform_IR<
                    processingStyle,
                    bitmap_f<>,
                    position_list_f<>
                >(plCol_2_cast);

        // reset cache before measurement
        clear_cache();

        auto intersect_bm_start = high_resolution_clock::now();

        // intersection:
        auto intersect_bm =
                intersect_sorted<
                    processingStyle,
                    bitmap_f<uncompr_f>,
                    bitmap_f<uncompr_f>,
                    bitmap_f<uncompr_f>
                >( bmCol_1, bmCol_2 );

        auto intersect_bm_end = high_resolution_clock::now();
        auto intersect_bm_exec_time = duration_cast<microseconds>(intersect_bm_end - intersect_bm_start);

        // store result:
        intersect_bm_avx512.insert({selectivity, intersect_bm_exec_time});
        (void)intersect_bm; // satisfy compiler error

        // reset cache before measurement
        clear_cache();

        auto union_bm_start = high_resolution_clock::now();

        // union:
        auto union_bm =
                merge_sorted<
                    processingStyle,
                    bitmap_f<uncompr_f>,
                    bitmap_f<uncompr_f>,
                    bitmap_f<uncompr_f>
                >( bmCol_1, bmCol_2 );

        auto union_bm_end = high_resolution_clock::now();
        auto union_bm_exec_time = duration_cast<microseconds>(union_bm_end - union_bm_start);

        // store result:
        merge_bm_avx512.insert({selectivity, union_bm_exec_time});
        (void)union_bm; // satisfy compiler error
    }

    // --------------- Write results to file ---------------
    std::ofstream mapStream;
    mapStream.open("micro_benchmark_4_uniform_sorted_unique_avx512.csv");

    // pl-intersect
    mapStream << "\"PL_intersection:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : intersect_pl_avx512){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfPlIntersectionResults\"\n";

    // bm-intersect
    mapStream << "\"BM_intersection:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : intersect_bm_avx512){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfBmIntersectionResults\"\n";

    // pl-merge
    mapStream << "\"PL_merge:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : merge_pl_avx512){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfPlMergeResults\"\n";

    // bm-merge
    mapStream << "\"BM_merge:\"" << "\n";
    mapStream << "\"selectivity\",\"execution time (μs)\"" << "\n";
    for(auto& element : merge_bm_avx512){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfBmMergeResults\"\n";
    mapStream.close();

    return 0;
}

