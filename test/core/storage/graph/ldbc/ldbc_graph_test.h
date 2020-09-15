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
 * @file ldbc_graph_test.cpp
 * @brief Test for generating social network graph in a given graph format
 * @todo
 */

#include <core/storage/graph/formats/adjacencylist.h>
#include <core/storage/graph/importer/ldbc_import.h>

void print_header(std::string storageFormat) {
    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Storage-Test: LDBC " << storageFormat << " Storage Format *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
}

template <class GRAPH_FORMAT> void ldbcGraphFormatTest(void) {

    static_assert(std::is_base_of<morphstore::Graph, GRAPH_FORMAT>::value,
                  "type parameter of this method must be a graph format");
                  
#ifdef LDBC_DIR
    std::unique_ptr<morphstore::Graph> graph = std::make_unique<GRAPH_FORMAT>();

    std::string storageFormat = graph->get_storage_format();

    print_header(storageFormat);

    std::unique_ptr<morphstore::LDBCImport> ldbcImport = std::make_unique<morphstore::LDBCImport>(LDBC_DIR);

    // generate vertices & edges from LDBC files and insert into graph structure
    ldbcImport->import(*graph);
    graph->statistics();

    graph->print_vertex_by_id(1035174);
    graph->print_edge_by_id(10);
    graph->print_neighbors_of_vertex(1035174);

    graph->morph(morphstore::GraphCompressionFormat::DELTA);

    graph->statistics();

    graph->print_vertex_by_id(1035174);
    graph->print_edge_by_id(10);
    graph->print_neighbors_of_vertex(1035174);

    // DEBUGGING
    //for(uint64_t id = 0; id < graph->getEdgeCount(); id++) {
    //    graph->get_outgoing_edge_ids(id);
    //}

    // measure degree distribution and write to file (file path as parameter):
    // TODO: but this into benchmark or so .. not actual test
    // std::cout << "Measure degree count" << std::endl;
    // graph->measure_degree_count(targetDir + "graph_degree_count_" + storageFormat + "SF1.csv");
#else
        throw std::invalid_argument("You forgot to define/uncomment the LDBC_DIR (at CMakeList.txt)"); 
#endif
}