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
 * @file graph_compression_benchmark.cpp
 * @brief A benchmark of the csr-graph compression (using the ldbc graph)
 * @todo allow different compression formats for the two csr columns; add full_iterate
 */

#include "benchmark_helper.h"
#include <core/storage/graph/formats/adjacencylist.h>
#include <core/storage/graph/importer/ldbc_import.h>
#include <random>

using namespace morphstore;

struct CompressionBenchmarkEntry {
    GraphCompressionFormat compr_format;
    uint64_t min_compr_degree;
    int64_t compression_time;
    double compression_ratio;
    double column_ratio;
    int64_t random_access_time;

    std::string to_string() {
        return "|" + graph_compr_f_to_string(compr_format) + "|" + std::to_string(min_compr_degree) + "|" +
               std::to_string(compression_time) + "|" + std::to_string(compression_ratio) + "|" +
               std::to_string(column_ratio) + "|" + std::to_string(random_access_time);
    }
};

int main(void) {
#ifdef LDBC_DIR
    // could be also build parameters?
    const int number_of_executions = 5;
    const int number_of_random_access = 1000;

    std::vector<GraphCompressionFormat> compr_formats = {GraphCompressionFormat::DELTA, GraphCompressionFormat::FOR,
                                                         GraphCompressionFormat::UNCOMPRESSED};

    std::vector<uint64_t> min_compr_degrees = {1024, 500, 100};

    // Load ldbc graph
    std::unique_ptr<AdjacencyList> graph = std::make_unique<AdjacencyList>();
    std::unique_ptr<LDBCImport> ldbcImport = std::make_unique<LDBCImport>(LDBC_DIR);
    ldbcImport->import(*graph);

    // prepare random-access
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist(0, graph->getVertexCount() - 1);
    std::vector<int> random_accesses;
    for (int i = 0; i < number_of_random_access; i++) {
        random_accesses.push_back(dist(rd));
    }

    std::cout << "Test vertex storage structure (median of 5 for full_iterate and random access)" << std::endl;
    std::cout << "Compression-Format | minimum degree for compression | compression-time | "
              << "compr. ratio | column ratio | access of edges of 5000 random vertices" << std::endl;

    for (auto min_compr_degree : min_compr_degrees) {
        for (auto current_f : compr_formats) {
            graph->set_min_compr_degree(min_compr_degree);

            for (int exec = 0; exec < number_of_executions; exec++) {
                CompressionBenchmarkEntry current_try;
                current_try.compr_format = current_f;
                current_try.min_compr_degree = graph->get_min_compr_degree();

                // restore start state
                graph->morph(GraphCompressionFormat::UNCOMPRESSED);

                auto start = highResClock::now();
                graph->morph(current_f);
                // compression time
                current_try.compression_time = get_duration(start);

                current_try.compression_ratio = graph->compr_ratio();
                // currently based on fixed min_compr_degree
                current_try.column_ratio = graph->column_ratio();

                // random access
                start = highResClock::now();
                for (int random_pos : random_accesses) {
                    graph->get_outgoing_edge_ids(random_pos);
                }
                current_try.random_access_time = get_duration(start);

                std::cout << current_try.to_string() << std::endl;
            }
        }
    }

    return 0;
#else
    throw std::invalid_argument("Where are the ldbc files??");
#endif
}
