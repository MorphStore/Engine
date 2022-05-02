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
 * @file micro_benchmark_2_scalar.h
 * @brief Experimental Evaluation:
 *              (2) WAH Compression / Decompression:
 *                  - Base data: uniform distribution with values 0 and TEST_DATA_COUNT-1
 *                  - WAH Input: uncompressed (verbatim) bitmap with different bit densities (using select-operator)
 *                  - WAH Output: compressed bitmap
 *                  - Compare Compression / Decompression with execution time + memory footprint
 *                  - Results are written to: micro_benchmark_2_uniform_scalar.csv
 */

// This must be included first to allow compilation.
#include <core/memory/mm_glob.h>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/morphing/wah.h>
#include <core/utils/printing.h>

#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>

#include <core/operators/general_vectorized/select_bm_uncompr.h>
#include <core/operators/general_vectorized/select_pl_uncompr.h>

#include <vector>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <cmath>

// local:
//#define TEST_DATA_COUNT 1000 * 10

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

    // hash map to store measurements: key = bit density; value = pair of {execution time, pair of{uncompressed_size, compressed_size}}
    std::unordered_map<double, std::pair<std::chrono::microseconds , std::pair<size_t, size_t>>> wah_compression_results;
    std::unordered_map<double, std::chrono::microseconds> wah_decompression_results;

    // --------------- (1) Generate test data ---------------
    auto inCol = generate_with_distr(
            TEST_DATA_COUNT,
            std::uniform_int_distribution<uint64_t>(
                    0,
                    TEST_DATA_COUNT - 1
            ),
            false
    );

    // for each i-th data point in TEST_DATA_COUNT: exec. less-than selection, calculate bit density + store measurement results
    //size_t steps = 100;
    // server:
    size_t steps = 1000000;
    for(auto i = 0; i < TEST_DATA_COUNT+1; i += steps){

        // --------------- (2) Selection operation ---------------
        // get uncompressed bitmap from SELECT-operator using avx2 (on server, use avx512 to speed up selection)
        auto bm_uncompr =
                morphstore::select<
                    less,
                    avx2<v256<uint64_t>>,
                    bitmap_f<uncompr_f>,
                    uncompr_f
                >(inCol, i);
        // calculate uncompressed memory footprint
        auto bm_uncompr_used_bytes = bm_uncompr->get_size_used_byte();

        // Unfortunately, we do not have any metadata yet to calculate the number of set bits for bit density
        // => Workaround: execute the selection with position-list to get the number of resulting elements
        auto pl_uncompr =
                morphstore::select<
                    less,
                    avx2<v256<uint64_t>>,
                    position_list_f<uncompr_f>,
                    uncompr_f
                >(inCol, i);
        // calculate bit density + round up to 2 dec. places (between 0 ... 1)
        double bit_density = std::ceil(
                (static_cast<double>(pl_uncompr->get_count_values()) / static_cast<double>(TEST_DATA_COUNT))
                * 100.0) / 100.0;

        // --------------- (3) WAH-Compression ---------------

        // clear cache before measurement
        clear_cache();

        auto bm_compr_start = high_resolution_clock::now();

        // compress bitmap
        auto bm_compr =
                morph_t<
                    processingStyle,
                    bitmap_f<wah_f>,
                    bitmap_f<uncompr_f>
                >::apply(bm_uncompr);

        auto bm_compr_end = high_resolution_clock::now();
        auto bm_compr_exec_time = duration_cast<microseconds>(bm_compr_end - bm_compr_start);

        // calculate compressed output
        auto bm_compr_used_bytes = bm_compr->get_size_used_byte();
        // compression ratio = uncompressed_size / compressed_size => round up to 3 dec. places
        /*double compression_ratio = std::ceil(
                (static_cast<double>(bm_uncompr_used_bytes) / static_cast<double>(bm_compr_used_bytes))
                * 1000.0) / 1000.0;
        */
        // store results to hash map
        if(wah_compression_results.count(bit_density) == 0) {
            wah_compression_results.insert({bit_density, {bm_compr_exec_time, {bm_uncompr_used_bytes, bm_compr_used_bytes}}});
        }

        // --------------- (4) WAH-Decompression ---------------

        // clear cache before measurement
        clear_cache();

        auto bm_decompr_start = high_resolution_clock::now();

        // decompress WAH-bitmap
        auto bm_decompr =
                morph_t<
                    processingStyle,
                    bitmap_f<uncompr_f>,
                    bitmap_f<wah_f>
                >::apply(bm_compr);

        auto bm_decompr_end = high_resolution_clock::now();
        auto bm_decompr_exec_time = duration_cast<microseconds>(bm_decompr_end - bm_decompr_start);

        // store results to hash map
        if(wah_decompression_results.count(bit_density) == 0) {
            wah_decompression_results.insert({bit_density, bm_decompr_exec_time});
        }
        (void)bm_decompr; // to satisfy compiler error 'unused variable'
    }

    // --------------- (5) Write results to file ---------------

    std::ofstream mapStream;
    mapStream.open("micro_benchmark_2_uniform_scalar.csv");

    mapStream << "\"WAH-Compression:\"" << "\n";
    mapStream << "\"bit density\",\"execution time (μs)\",\"uncompressed_size (B)\",\"compressed_size (B)\"" << "\n";
    for(auto& element : wah_compression_results){
        mapStream << element.first
                  << "," << element.second.first.count()
                  << "," << element.second.second.first
                  << "," << element.second.second.second
                  << "\n";
    }
    mapStream << "\"endOfWahCompressionResults\"\n";

    mapStream << "\"WAH-Decompression:\"" << "\n";
    mapStream << "\"bit density\",\"execution time (μs)\"" << "\n";
    for(auto& element : wah_decompression_results){
        mapStream << element.first
                  << "," << element.second.count()
                  << "\n";
    }
    mapStream << "\"endOfWahDecompressionResults\"\n";

    mapStream.close();

    return 0;
}