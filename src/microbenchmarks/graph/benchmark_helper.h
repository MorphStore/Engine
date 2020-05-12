/**********************************************************************************************
 * Copyright (C) 2020 by MorphStore-Team                                                      *
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
 * @file benchmark_helper.h
 * @brief Helper functions for graph benchmarks
 * @todo 
*/

#ifndef BENCHMARK_HELPER
#define BENCHMARK_HELPER

#include <chrono>
#include <algorithm>
#include <assert.h>

namespace morphstore {
    using highResClock = std::chrono::high_resolution_clock;

    int64_t get_duration(std::chrono::time_point<std::chrono::system_clock> start) {
        auto stop = highResClock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    }

    int64_t get_median(std::vector<int64_t> values) {
        assert(values.size() > 0);
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    }
} // namespace morphstore

#endif //BENCHMARK_HELPER
