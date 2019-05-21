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

#include <core/storage/graph/graph.h>
#include <core/storage/graph/ldbc_import.h>

int main( void ){

    // ------------------------------------ LDBC-IMPORT TEST ------------------------------------

    // TODO: get base directory with cin -> user input
    morphstore::LDBC_Import ldbcImport("/home/tim/ldbc_snb_datagen-0.2.8/social_network/");
    morphstore::Graph socialGraph;

    // generate vertices & edges from LDBC files and insert into socialGraph
    ldbcImport.generate_vertices(socialGraph);
    ldbcImport.generate_edges(socialGraph);

    socialGraph.statistics();
    /*
    // test vertices:
    socialGraph.print_vertex_by_id(100454);
    socialGraph.print_vertex_by_id(100450);
    socialGraph.print_vertex_by_id(100168);
    */
    return 0;
}