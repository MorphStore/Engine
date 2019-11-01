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
 * @file ldbc_graph_adjacency.cpp
 * @brief Test for generating social network graph in adj. list format + BFS measurements
 * @todo
 */

#include <core/storage/graph/ldbc_import.h>
#include <core/storage/graph/formats/adjacencylist.h>
#include <core/operators/graph/top_down_bfs.h>

#include <chrono>  // for high_resolution_clock

int main( void ){

    // ------------------------------------ LDBC-IMPORT TEST -----------------------------------
    /*
    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: Adjacency-List Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
    */

    // ldbc importer: path to csv files as parameter: (don't forget the last '/' in adress path)
    std::unique_ptr<morphstore::LDBCImport> ldbcImport = std::make_unique<morphstore::LDBCImport>("/home/pfeiffer/ldbc_sn_data/social_network_1/");

    // Graph init:
    std::unique_ptr<morphstore::Graph> g1 = std::make_unique<morphstore::AdjacencyList>();

    // generate vertices & edges from LDBC files and insert into graph structure
    ldbcImport->import(*g1);

    // measure degree distribution and write to file (file path as parameter):
    g1->measure_degree_count("/home/pfeiffer/measurements/adjacency_list/graph_degree_count_SF10.csv");

    // some statistics (DEBUG)
    // g1->statistics();

    // (DEBUG) Test Vertex, which contains edges with properties (SERVER):
    // g1->print_vertex_by_id(1035174);
    // g1->print_neighbors_of_vertex(1035174);

    // Execute BFS measurements:
    // std::unique_ptr<morphstore::BFS> bfs = std::make_unique<morphstore::BFS>(g1);
    // bfs->do_measurements(10000, "/home/pfeiffer/measurements/adjacency_list/bfs_SF1.csv");

    return 0;
}
