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
 * @todo TODOS?
 */

#include <core/storage/graph/graph.h>
#include <core/storage/graph/ldbc_import.h>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <stdio.h>

using namespace std;

/*
void generateEdges(vector<Relation>& rDict, morphstore::Graph& g){

    cout << "Generating Relations ...";
    std::cout.flush();

    // iterate through relationDict and add (target.id, rel.id) to the vertex adj.-list
    for(std::vector<Relation>::iterator it = rDict.begin(); it != rDict.end(); ++it){
        g.add_edge(it->fromID, it->toID, it->relID);
    }

    cout << " --> done" << endl;
}

 */

int main( void ){

    // ------------------------------------ LDBC-IMPORT TEST ------------------------------------

    // TODO: get base directory with cin -> user input
    morphstore::LDBC_Import ldbcImport("/home/tim/ldbc_snb_datagen-0.2.8/social_network/");
    morphstore::Graph socialGraph;

    //ldbcImport.print_file_names();
    ldbcImport.generate_vertices(socialGraph);
    socialGraph.statistics();

    return 0;
}