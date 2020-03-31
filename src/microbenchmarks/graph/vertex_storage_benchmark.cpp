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



int main(void) {
    using namespace morphstore;

    for(int vertex_count=10000; vertex_count < 100000000; vertex_count = vertex_count*10) {
        std::cout << "Testing graph with vertex count of:" << vertex_count << std::endl;
        std::unique_ptr<CSR> graph = std::make_unique<CSR>();
        graph->allocate_graph_structure(vertex_count, 0);
        for(int i=0; i < vertex_count; i++) {
            graph->add_vertex(i);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // iterate
        for(int i=0; i < vertex_count; i++) {
            graph->get_vertex(i);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto iteration_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << "Iteration time: " << iteration_duration << "ms" << std::endl;
        // random access
        std::random_device rd;
        std::uniform_int_distribution<uint64_t> dist(0, vertex_count);

        start = std::chrono::high_resolution_clock::now();

        for(int i=0; i < 10000; i++) {
            graph->get_vertex(dist(rd));
        }

        stop = std::chrono::high_resolution_clock::now();
        auto random_access_duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
        std::cout << "Random access of 10000 vertices: " << random_access_duration << "ms" << std::endl;
    }

    return 0;
}
