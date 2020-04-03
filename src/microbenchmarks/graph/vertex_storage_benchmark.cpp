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
 * @file vertex_storage_benchmark.cpp
 * @brief A little mirco benchmark of the vertex storage (hashmap<id, vertex> vs vector<vector<vertex>>).
 */

#include <core/storage/graph/formats/csr.h>
#include <chrono>
#include <random>


typedef std::chrono::high_resolution_clock highResClock;
using namespace morphstore;

int64_t getDuration(std::chrono::time_point<std::chrono::system_clock> start) {
    auto stop = highResClock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}

int main(void) {
    // TODO: use core/utils/monitoring.h ? or a "time_it" function to stop a given function

    int number_of_executions = 5;

    std::cout << "Test vertex storage structure (avg of 5 for full_iterate and random access) times in μs" << std::endl;
    std::cout << "vertex_count | loading time | full_iterate | 10^4 random access" << std::endl;

    for(int vertex_count=10000; vertex_count < 100000000; vertex_count = vertex_count*10) {
        int64_t duration = 0;
        
        std::cout << vertex_count << " | ";
        std::unique_ptr<CSR> graph = std::make_unique<CSR>();
        graph->allocate_graph_structure(vertex_count, 0);
        auto start = highResClock::now();
        for(int i=0; i < vertex_count; i++) {
            graph->add_vertex(i);
        }

        std::cout << getDuration(start)  << " | ";
        
        duration = 0;

        for(int exec=0; exec < number_of_executions;  exec++) {
            auto start = highResClock::now();
            // iterate
            for(int i=0; i < vertex_count; i++) {
                graph->get_vertex(i);
            }
            duration += getDuration(start);
        }

        std::cout << duration / number_of_executions << " | ";


        // random access

        duration = 0;

        for(int exec=0; exec < number_of_executions;  exec++) { 
            std::random_device rd;
            std::uniform_int_distribution<uint64_t> dist(0, vertex_count - 1);

            auto start = highResClock::now();

            for(int i=0; i < 10000; i++) {
                graph->get_vertex(dist(rd));
            }

            duration += getDuration(start);
        }

        std::cout << duration / number_of_executions << std::endl;
    }

    return 0;
}
