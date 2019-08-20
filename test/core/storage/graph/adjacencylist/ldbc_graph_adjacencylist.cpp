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
 * @brief Test for generating social network graph in ADJ_LIST format
 * @todo
 */

#include <core/storage/graph/ldbc_import.h>
#include <core/storage/graph/formats/adjacencylist.h>
#include <chrono>  // for high_resolution_clock

int main( void ){

    // ------------------------------------ LDBC-IMPORT TEST -----------------------------------
    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: Adjacency-List Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";

    auto start = std::chrono::high_resolution_clock::now(); // For measuring the execution time

    morphstore::LDBCImport ldbcImport("/opt/ldbc_snb_datagen-0.2.8/social_network/");

    // Graph init:
    std::unique_ptr<morphstore::Graph> g1 = std::make_unique<morphstore::AdjacencyList>();

    // generate vertices & edges from LDBC files and insert into graph
    ldbcImport.import(g1);

    // measuring time:
    auto finish = std::chrono::high_resolution_clock::now(); // For measuring the execution time
    std::chrono::duration<double> elapsed = finish - start;

    g1->statistics();
    std::cout << "Import & Graph-Generation Time: " << elapsed.count() << " sec.\n";

    /*
    // test vertices:
    g1->print_vertex_by_id(100454);
    g1->print_vertex_by_id(100450);
    g1->print_vertex_by_id(100168);
    g1->print_vertex_by_id(2000100);
     */

    // calculate size of social graph
    //std::cout << "Size of social network: " << socialGraph.get_size_of_graph() << " Bytes\n";

    return 0;
}