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
 * @file bfs_csr_partial_compression_benchmark.cpp
 * @brief A benchmark evaluating the impact of graph compression on breadth first search (using the ldbc graph) and
 * only compressing on csr column
 * @todo cleanup benchmark (this was only created last minute)
 */

#include "benchmark_helper.h"
#include <core/operators/graph/top_down_bfs.h>
#include <core/storage/graph/formats/csr.h>
#include <core/storage/graph/importer/ldbc_import.h>

#include <filesystem>

using namespace morphstore;

struct CompressionBenchmarkEntry {
    std::string graph_format;
    std::string compr_format;
    int64_t bfs_time;
    int64_t visited_vertices;

    std::string to_string() {
        return graph_format + "|" + compr_format + "|" + std::to_string(bfs_time) + "|" +
               std::to_string(visited_vertices);
    }
};

int main(void) {
#ifdef LDBC_DIR
    // could be also build parameters?
    const int number_of_executions = 5;
    const int number_of_start_vertices = 10;

    // combination of uncompress + other
    std::vector<std::pair<GraphCompressionFormat, GraphCompressionFormat>> compr_formats = {
        {GraphCompressionFormat::DELTA, GraphCompressionFormat::UNCOMPRESSED},
        {GraphCompressionFormat::FOR, GraphCompressionFormat::UNCOMPRESSED},
        {GraphCompressionFormat::DYNAMIC_VBP, GraphCompressionFormat::UNCOMPRESSED},
        {GraphCompressionFormat::UNCOMPRESSED, GraphCompressionFormat::DELTA},
        {GraphCompressionFormat::UNCOMPRESSED, GraphCompressionFormat::FOR},
        {GraphCompressionFormat::UNCOMPRESSED, GraphCompressionFormat::DYNAMIC_VBP}};

    // Load ldbc graph
    // blank lines for easier deletion of progress prints
    std::cout << std::endl << std::endl;
    std::shared_ptr<CSR> graph = std::make_shared<CSR>();
    std::unique_ptr<LDBCImport> ldbcImport = std::make_unique<LDBCImport>(LDBC_DIR);
    ldbcImport->import(*graph);
    std::cout << std::endl << std::endl;

    const int cycle_size = graph->getVertexCount() / number_of_start_vertices;
    auto start_vertex_ids = BFS::get_list_of_every_ith_vertex(graph, cycle_size);

    // BFS
    std::cout
        << "Test impact of compression on BFS (10 start-nodes (evenly distributed regarding degree); 5x excutions)"
        << std::endl;
    std::cout << "Graph-Format | Compression-Format | bfs-time in micro seconds| visited vertices" << std::endl;

    for (auto [offset_format, edgeId_format] : compr_formats) {
        for (int exec = 0; exec < number_of_executions; exec++) {
            CompressionBenchmarkEntry current_try;
            current_try.graph_format = graph->get_storage_format();

            if(offset_format != GraphCompressionFormat::UNCOMPRESSED) {
                current_try.compr_format = graph_compr_f_to_string(offset_format);
                current_try.graph_format += "(only offsets compressed)";
            }
            else if (edgeId_format != GraphCompressionFormat::UNCOMPRESSED) {
                current_try.compr_format = graph_compr_f_to_string(edgeId_format);
                current_try.graph_format += "(only edgeIds compressed)"; 
            }

            // restore start state (not needed as this will be not timed and morphing internally goes via uncompr)
            // graph->morph(GraphCompressionFormat::UNCOMPRESSED, false);
            // morphing into desired format
            graph->morph(offset_format, edgeId_format);

            for (auto id : start_vertex_ids) {
                auto start = highResClock::now();
                current_try.visited_vertices = morphstore::BFS::compute(graph, id);
                current_try.bfs_time = get_duration(start);

                // for saving into csv file, just use "> xyz.csv" at execution
                std::cout << current_try.to_string() << std::endl;
            }
        }
    }
#else
    throw std::invalid_argument("You forgot to define/uncomment the LDBC_DIR (at CMakeList.txt)");
#endif
}
