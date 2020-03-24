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

#include <core/storage/graph/formats/csr.h>
#include <assert.h>
//#include <core/operators/graph/top_down_bfs.h>

int main( void ){
    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: CSR-List Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";

    // Graph init:
    std::unique_ptr<morphstore::Graph> g1 = std::make_unique<morphstore::CSR>();
    g1->allocate_graph_structure(3, 3);

    std::map<unsigned short, std::string> edgeTypeMap = {{1, "knows"}, {2, "likes"}}; 
    std::map<unsigned short, std::string> vertexTypeMap = {{0, "Person"}}; 
    g1->setEdgeTypeDictionary(edgeTypeMap);
    g1->setVertexTypeDictionary(vertexTypeMap);
    
    uint64_t v1 = g1->add_vertex(0, {{"age", "12"}});
    uint64_t v2 = g1->add_vertex(0);
    uint64_t v3 = g1->add_vertex(0);
    


    g1->add_edges(v1, {morphstore::Edge(v1, v2, 1, {{"rating", "42"}, {"description", "has the answer to everything"}})});
    g1->add_edges(v2, {morphstore::Edge(v2, v3, 2), morphstore::Edge(v2, v3, 1)});


    // (DEBUG)
    /*g1->statistics();
    g1->print_edge_by_id(0);
    g1->print_neighbors_of_vertex(v1);
    g1->print_neighbors_of_vertex(v2);
    g1->print_neighbors_of_vertex(v3);*/

    assert(g1->getVertexCount() == 3);
    assert(g1->getEdgeCount() == 3);
    assert((int) g1->get_edge(0)->getProperties().size() == 2);
    assert(g1->get_out_degree(v3) == 0);
    assert(g1->get_out_degree(v1) == 1);
    assert(g1->get_out_degree(v2) == 2);

    return 0;
}
