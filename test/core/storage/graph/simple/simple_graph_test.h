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
 * @file simple_graph_test.cpp
 * @brief Base test for testing graph formats on a very simple graph
 * @todo
 */
#include <core/storage/graph/graph.h>
#include <assert.h>

void print_header(std::string storageFormat) {
    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: Simple " << storageFormat << " Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
}

template <class GRAPH_FORMAT>
void simpleGraphFormatTest (void) {
    static_assert(std::is_base_of<morphstore::Graph, GRAPH_FORMAT>::value, "type parameter of this method must be a graph format");

    std::unique_ptr<morphstore::Graph> graph = std::make_unique<GRAPH_FORMAT>();
    print_header(graph->get_storage_format());

     graph->allocate_graph_structure(3, 3);

    std::map<unsigned short, std::string> edgeTypeMap = {{1, "knows"}, {2, "likes"}};
    std::map<unsigned short, std::string> vertexTypeMap = {{0, "Person"}};
    graph->setEdgeTypeDictionary(edgeTypeMap);
    graph->set_vertex_type_dictionary(vertexTypeMap);

    uint64_t v1 = graph->add_vertex(0, {{"age", "12"}});
    uint64_t v2 = graph->add_vertex(0);
    uint64_t v3 = graph->add_vertex(0);

    auto e1 = morphstore::Edge(v1, v2, 1);

    graph->add_edges(v1, {e1});
    graph->add_properties_to_edge(e1.getId(), {{"rating", 42}, {"description", "has the answer to everything"}});
    graph->add_edges(v2, {morphstore::Edge(v2, v3, 2), morphstore::Edge(v2, v3, 1)});

    // (DEBUG)
    graph->statistics();
    graph->print_edge_by_id(0);
    graph->compress(morphstore::GraphCompressionFormat::RLE);
    graph->print_neighbors_of_vertex(v2);
    graph->statistics();

    assert(graph->getVertexCount() == 3);
    assert(graph->getEdgeCount() == 3);
    assert((int)graph->get_edge(e1.getId()).getProperties().size() == 2);
    assert(graph->get_out_degree(v3) == 0);
    assert(graph->get_out_degree(v1) == 1);
    assert(graph->get_out_degree(v2) == 2);
}

