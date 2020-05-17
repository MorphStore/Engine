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
 * @file bfs_ldbc_graph_test.cpp
 * @brief Test methods for bfs on social network graph
 * @todo
 */

#include <core/operators/graph/top_down_bfs.h>
#include <core/storage/graph/formats/adjacencylist.h>
#include <core/storage/graph/importer/ldbc_import.h>

void print_header(std::string storageFormat) {

    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Operator-Test: LDBC " << storageFormat << " BFS Test *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
}

template <class GRAPH_FORMAT> void bfs_ldbc_graph_test(void) {
#ifdef LDBC_DIR
    static_assert(std::is_base_of<morphstore::Graph, GRAPH_FORMAT>::value,
                  "type parameter of this method must be a graph format");

    std::unique_ptr<morphstore::Graph> graph = std::make_unique<GRAPH_FORMAT>();
    std::string storageFormat = graph->get_storage_format();

    print_header(storageFormat);

    // ldbc importer: path to csv files as parameter: (don't forget the last '/' in adress path)
    std::unique_ptr<morphstore::LDBCImport> ldbcImport = std::make_unique<morphstore::LDBCImport>(LDBC_DIR);

    // generate vertices & edges from LDBC files and insert into graph structure
    ldbcImport->import(*graph);

    // some statistics (DEBUG)
    std::cout << "Some statistics" << std::endl;
    graph->statistics();

    auto bfs = std::make_unique<morphstore::BFS>(graph);
    // for scale factor 1 and including static as well as dynamic part of the graph
    std::cout << "Based on Vertex with id 0: " << bfs->do_BFS(0) << " vertices could be explored via BFS";
    // bfs->do_measurements(10000, targetDir + "bfs_" + storageFormat);
#else 
    throw std::invalid_argument("Where are the ldbc files??");
#endif
}