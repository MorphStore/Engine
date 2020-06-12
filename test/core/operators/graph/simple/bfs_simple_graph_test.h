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
 * @file bfs_simple_graph_test.cpp
 * @brief Test methods for bfs on social network graph
 * @todo
 */

#include <core/storage/graph/formats/adjacencylist.h>
#include <core/operators/graph/top_down_bfs.h>
#include <assert.h>

void print_header(std::string storageFormat) {

    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Operator-Test: Simple " << storageFormat << " BFS Test *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
}

template <class GRAPH_FORMAT>
void bfs_simple_graph_test (void) {

    static_assert(std::is_base_of<morphstore::Graph, GRAPH_FORMAT>::value, "type parameter of this method must be a graph format");

    std::shared_ptr<morphstore::Graph> graph = std::make_shared<GRAPH_FORMAT>();
    print_header(graph->get_storage_format());

    graph->allocate_graph_structure(4, 4);

    std::map<unsigned short, std::string> edgeTypeMap = {{1, "knows"}, {2, "likes"}};
    std::map<unsigned short, std::string> vertexTypeMap = {{0, "Person"}};
    graph->setEdgeTypeDictionary(edgeTypeMap);
    graph->set_vertex_type_dictionary(vertexTypeMap);

    uint64_t v1 = graph->add_vertex(0);
    uint64_t v2 = graph->add_vertex(0);
    uint64_t v3 = graph->add_vertex(0);
    graph->add_vertex(0);

    graph->add_edges(v1, {morphstore::Edge(v1, v2, 1)});
    graph->add_edges(v2, {morphstore::Edge(v2, v3, 2), morphstore::Edge(v2, v3, 1)});
    graph->add_edges(v3, {morphstore::Edge(v3, v2, 1)});
    // some statistics (DEBUG)
    std::cout << "Some statistics" << std::endl;
    graph->statistics();

    assert(graph->getVertexCount() == 4);
    assert(graph->getEdgeCount() == 4);
    
    assert(morphstore::BFS::compute(graph, v1) == 2);
    assert(morphstore::BFS::compute(graph, v2) == 1);
    assert(morphstore::BFS::compute(graph, v3) == 1);
}