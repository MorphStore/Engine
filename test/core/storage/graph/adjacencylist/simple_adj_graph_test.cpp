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
 * @file simple_graph_test_adj.cpp
 * @brief Test for generating simple graph in adj. list format (+ BFS measurements)
 * @todo
 */

#include <core/storage/graph/ldbc_import.h>
#include <core/storage/graph/formats/adjacencylist.h>
//#include <core/operators/graph/top_down_bfs.h>
//#include <chrono>  // for high_resolution_clock

int main( void ){

    // ------------------------------------ LDBC-IMPORT TEST -----------------------------------
    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: Adjacency-List Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";

    // Graph init:
    std::unique_ptr<morphstore::Graph> g1 = std::make_unique<morphstore::AdjacencyList>();

    // generate vertices & edges from LDBC files and insert into graph structure
    uint64_t v1 = g1->add_vertex_with_properties({{"age", "12"}});
    uint64_t v2 = g1->add_vertex();
    uint64_t v3 = g1->add_vertex();
    
    std::map<unsigned short, std::string> edgeTypeMap = {{1, "knows"}, {2, "likes"}}; 
    std::map<unsigned short, std::string> vertexTypeMap = {{0, "Person"}}; 
    g1->setEdgeTypeDictionary(edgeTypeMap);
    g1->setVertexTypeDictionary(vertexTypeMap);

    g1->add_edge(v1, v2, 1);
    g1->add_edge(v2, v3, 1);
    g1->add_edge(v2, v3, 2);


    // (DEBUG)
    g1->statistics();
    g1->print_edge_by_id(1);
    g1->print_neighbors_of_vertex(v1);
    g1->print_neighbors_of_vertex(v2);
    g1->print_neighbors_of_vertex(v3);

    return 0;
}
