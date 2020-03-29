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
 * @file bfs_ldbc_graph_test.cpp
 * @brief Test methods for bfs on social network graph
 * @todo
 */

#include <core/storage/graph/ldbc_import.h>
#include <core/storage/graph/formats/adjacencylist.h>
#include <core/operators/graph/top_down_bfs.h>

void print_header(morphstore::Graph::storageFormat format) {
    std::string storageFormat;

    switch (format)
    {
    case morphstore::Graph::storageFormat::csr:
        storageFormat = "CSR";
        break;
    case morphstore::Graph::storageFormat::adjacencylist:
        storageFormat = "Adjacency-List";
        break;
    };

    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: LDBC " << storageFormat << " Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
}

template <class GRAPH_FORMAT>
void bfs_ldbc_graph_test (void) {

    static_assert(std::is_base_of<morphstore::Graph, GRAPH_FORMAT>::value, "type parameter of this method must be a graph format");

    std::string sourceDir = "";
    std::string targetDir = "";

    if (sourceDir.empty()) {
        throw std::invalid_argument("Where are the ldbc files??");
    }

    if (targetDir.empty()) {
        throw std::invalid_argument("Degree count has to be saved somewhere");
    }

    std::unique_ptr<morphstore::Graph> graph = std::make_unique<GRAPH_FORMAT>();
    print_header(graph->getStorageFormat());

    // ldbc importer: path to csv files as parameter: (don't forget the last '/' in adress path)
    std::unique_ptr<morphstore::LDBCImport> ldbcImport = std::make_unique<morphstore::LDBCImport>(sourceDir);


    // generate vertices & edges from LDBC files and insert into graph structure
    ldbcImport->import(*graph);

    // some statistics (DEBUG)
    std::cout << "Some statistics" << std::endl;
    graph->statistics();

    auto bfs = std::make_unique<morphstore::BFS>(graph);

    bfs->do_measurements(10000, targetDir + "bfs_" + graph->get_storage_format_string());
}