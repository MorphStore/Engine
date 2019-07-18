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
 * @file generate_ldbc_graph.cpp
 * @brief Test for generating social network graph from LDBC files
 * @todo
 */

#include <core/storage/graph/csr/graph.h>
#include <core/storage/graph/ldbc_import.h>
#include <core/storage/graph/graph_abstract.h>
#include <chrono>  // for high_resolution_clock

int main( void ){

    // ------------------------------------ LDBC-IMPORT TEST ------------------------------------

    morphstore::LDBC_Import ldbcImport("/opt/ldbc_snb_datagen-0.2.8/social_network/");
    morphstore::CSR socialGraph;

    // create abstract pointer to adjc_list (ldbc importer just has to handle with one input class and not adjcancyList, CSR, ....)
    morphstore::Graph *graph;
    graph = &socialGraph;

    ldbcImport.generate_vertices(*graph);

    socialGraph.statistics();

    std::cout << "Number of edges: " << ldbcImport.get_total_number_vertices() << std::endl;
    std::cout << "Number of edges: " << ldbcImport.get_total_number_edges() << std::endl;

    return 0;
}