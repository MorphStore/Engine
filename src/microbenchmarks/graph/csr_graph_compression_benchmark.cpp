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
 * @file csr_graph_compression_benchmark.cpp
 * @brief A benchmark of the csr-graph compression (using the ldbc graph)
 * @todo allow different compression formats for the two csr columns; add full_iterate
 */

#include <core/storage/graph/formats/csr.h>
#include <core/storage/graph/importer/ldbc_import.h>
#include <random>
#include "benchmark_helper.h"

using namespace morphstore;


struct CompressionBenchmarkEntry {
  GraphCompressionFormat compr_format;
  int64_t compression_time;
  double offset_col_compression_ratio;
  double edgeId_col_compression_ratio;
  int64_t random_access_time;
  int64_t full_iterate;

  std::string to_string() {
      return "|" + graph_compr_f_to_string(compr_format) +
             "|" + std::to_string(compression_time) +
             "|" + std::to_string(offset_col_compression_ratio) +
             "|" + std::to_string(edgeId_col_compression_ratio) +
             "|" + std::to_string(random_access_time);
  }
};

int main(void) {
    // could be also build parameters?
    const int number_of_executions = 5;
    const int number_of_random_access = 1000;
    std::string sourceDir = "";

    if (sourceDir.empty()) {
        throw std::invalid_argument("Where are the ldbc files??");
    }


    std::vector<GraphCompressionFormat> compr_formats = {
        GraphCompressionFormat::UNCOMPRESSED,
        GraphCompressionFormat::DELTA, 
        GraphCompressionFormat::FOR 
        };
    
    // Load ldbc graph
    std::unique_ptr<CSR> graph = std::make_unique<CSR>();
    std::unique_ptr<LDBCImport> ldbcImport = std::make_unique<LDBCImport>(sourceDir);
    ldbcImport->import(*graph);

    // prepare random-access
    std::random_device rd;
    std::uniform_int_distribution<uint64_t> dist(0, graph->getVertexCount() - 1);
    std::vector<int> random_accesses;
    for (int i = 0; i < number_of_random_access; i++) {
        random_accesses.push_back(dist(rd));
    }


    std::cout << "Test compression of ldbc-graph in CSR format (times in micro-seconds)" << std::endl;
    std::cout << "Compression-Format | compression-time | offset-column compr. ratio" <<
                     " | edgeId-column compr. ratio | access of edges of " <<
                     std::to_string(number_of_random_access) + " random vertices | full edge-list iterate"
              << std::endl;

    for (auto current_f : compr_formats) {
      for (int exec = 0; exec < number_of_executions; exec++) {
        CompressionBenchmarkEntry current_try;
        current_try.compr_format = current_f;
        // restore start state
        graph->morph(GraphCompressionFormat::UNCOMPRESSED);

        auto start = highResClock::now();
        graph->morph(current_f);
        // compression time 
        current_try.compression_time = get_duration(start);

        // compression-ratios
        current_try.offset_col_compression_ratio = graph->offset_column_compr_ratio();
        current_try.edgeId_col_compression_ratio = graph->edgeId_column_compr_ratio();


        // random access
        start = highResClock::now();
        for (int random_pos : random_accesses) {
            graph->get_outgoing_edge_ids(random_pos);
        }
        current_try.random_access_time = get_duration(start);

        std::cout << current_try.to_string() << std::endl;
      }
    }

    return 0;
}
