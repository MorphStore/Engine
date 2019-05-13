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

#include <core/storage/graph.h>

#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <stdio.h>

using namespace std;

struct Relation{
    uint64_t fromID;
    uint64_t toID;
    int relID;
};

void importDataLookup(string address, unordered_map<int, string> &rLookup){
    cout << "Reading LDBC-Lookups ...";
    std::cout.flush();

    char* buffer;
    ifstream data(address, std::ios::binary | std::ios::ate ); // 'ate' means: open and seek to end immediately after opening
    uint64_t fileSize = 0;

    if(!data){
        cerr << "\nError, opening file. ";
        exit(EXIT_FAILURE);
    }

    if (data.is_open()) {
        fileSize = data.tellg(); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
        data.clear();
        data.seekg( 0, std::ios::beg ); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
    }

    // allocate memory with the filesize and the char size
    buffer = (char*) malloc( fileSize * sizeof( char ) );
    data.read(buffer, fileSize); // read data as one big block
    size_t start = 0;
    string delimiter = "\t";

    for(size_t i = 0; i < fileSize; ++i){
        if(buffer[i] == '\n'){

            // get a row into string form buffer with start- and end-point and do stuff ...
            string row(&buffer[start], &buffer[i]);

            // remove unnecessary '\n' at the beginning of a string
            if(row.find('\n') != string::npos){
                row.erase(0,1);
            }

            string relationName = row.substr(0, row.find(delimiter));
            row.erase(0, row.find(delimiter) + delimiter.length());
            string relID_str = row.substr(0, row.find(delimiter));

            // convert string data to needed types
            int relID = stoi(relID_str, nullptr, 10);

            // put into lookup data structure
            rLookup.insert(make_pair(relID, relationName));

            start = i; // set new starting point (otherwise it's concatenated)
        }
    }

    delete[] buffer; // free memory
    data.close();

    cout << " --> done" << endl;
}

void importDataVertex(string vertexFile, unordered_map<uint64_t , pair<int, uint64_t>> &vDict, unordered_map<int, string> &eLookup){

    cout << "Reading LDBC-Vertices ...";
    std::cout.flush();

    char* buffer;
    ifstream graph(vertexFile, std::ios::binary | std::ios::ate ); // 'ate' means: open and seek to end immediately after opening
    uint64_t fileSize = 0;

    if(!graph){
        cerr << "Error, opening file. ";
        exit(EXIT_FAILURE);
    }

    if (graph.is_open()) {
        fileSize = graph.tellg(); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
        graph.clear();
        graph.seekg( 0, std::ios::beg ); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
    }

    // allocate memory with the filesize and the char size
    buffer = (char*) malloc( fileSize * sizeof( char ) );
    graph.read(buffer, fileSize); // read data as one big block
    size_t start = 0;
    string delimiter = "\t";
    int entityIndex = 0;

    for(size_t i = 0; i < fileSize; ++i){
        if(buffer[i] == '\n'){
            // get a row into string form buffer with start- and end-point and do stuff ...
            string row(&buffer[start], &buffer[i]);

            // remove unnecessary '\n' at the beginning of a string
            if(row.find('\n') != string::npos){
                row.erase(0,1);
            }

            // for entities we have to look that there is NO '\t' in the string
            if(row.find(delimiter) == string::npos){
                eLookup.insert(make_pair(entityIndex, row));
                entityIndex++;
            }else{

                // acutal data: first is ldbc_id and second global_id
                string ldbc_str = row.substr(0, row.find(delimiter));
                string global_str = row.erase(0, row.find(delimiter) + delimiter.length()); // erase from row (...)\t[data]

                // convert string to long int // TODO: is stoul enough for string -> size_t
                uint64_t ldbc_id = stoul(ldbc_str,nullptr,10);
                uint64_t global_id = stoul(global_str,nullptr,10);

                vDict.insert({global_id, make_pair(entityIndex-1, ldbc_id)});

            }
            start = i; // set new starting point (otherwise it's concatenated)
        }
    }

    delete[] buffer; // free memory
    graph.close();

    cout << " --> done" << endl;
}

void importDataRelations(string relationsFile, vector<Relation> &rList){

    cout << "Reading LDBC-Relations ...";
    std::cout.flush();

    char* buffer;
    ifstream graph(relationsFile, std::ios::binary | std::ios::ate ); // 'ate' means: open and seek to end immediately after opening
    uint64_t fileSize = 0;

    if(!graph){
        cerr << "Error, opening file. ";
        exit(EXIT_FAILURE);
    }

    if (graph.is_open()) {
        fileSize = graph.tellg(); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
        graph.clear();
        graph.seekg( 0, std::ios::beg ); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
    }

    // allocate memory with the filesize and the char size
    buffer = (char*) malloc( fileSize * sizeof( char ) );
    graph.read(buffer, fileSize); // read data as one big block
    size_t start = 0;
    string delimiter = "\t";

    for(size_t i = 0; i < fileSize; ++i){
        if(buffer[i] == '\n'){

            // get a row into string form buffer with start- and end-point and do stuff ...
            string row(&buffer[start], &buffer[i]);

            string fromID_str = row.substr(0, row.find(delimiter));
            row.erase(0, row.find(delimiter) + delimiter.length());
            string toID_str = row.substr(0, row.find(delimiter));
            string relID_str = row.erase(0, row.find(delimiter) + delimiter.length());

            // convert string data to needed types
            uint64_t fromID = stoul(fromID_str,nullptr,10);
            if(toID_str == "-1") toID_str = fromID_str; // if the toID is -1 --> loop to itself; refers to the multiple attributes
            uint64_t toID = stoul(toID_str,nullptr,10);
            int relID = stoi(relID_str, nullptr, 10);

            // write to relationDict data structure
            Relation r;
            r.fromID = fromID;
            r.toID = toID;
            r.relID = relID;
            rList.push_back(r);

            start = i; // set new starting point (otherwise it's concatenated)
        }
    }

    delete[] buffer; // free memory
    graph.close();

    cout << " --> done" << endl;
}

void generateVertices(unordered_map<uint64_t, pair<int, uint64_t>>& vertexDict, morphstore::Graph& g){

    cout << "Generating Vertices ...";
    std::cout.flush();

    // iterate through vertex-dict. and generate the vertices (objects) in the graph
    for(std::unordered_map<uint64_t, pair<int, uint64_t>>::iterator it = vertexDict.begin(); it != vertexDict.end(); ++it){
        uint64_t id = it->first;
        uint64_t ldbc_id = it->second.second;
        int entity = it->second.first;
        g.addVertex(id, ldbc_id, entity);
    }

    cout << " --> done" << endl;
}

void generateEdges(vector<Relation>& rDict, morphstore::Graph& g){

    cout << "Generating Relations ...";
    std::cout.flush();

    // iterate through relationDict and add (target.id, rel.id) to the vertex adj.-list
    for(std::vector<Relation>::iterator it = rDict.begin(); it != rDict.end(); ++it){
        g.addEdge(it->fromID, it->toID, it->relID);
    }

    cout << " --> done" << endl;
}

int main( void ){

    // -------------------------------- Reading data from LDBC-tsv-files --------------------------------

    // TODO: change intermediate results[] tsv -> [dicts] -> vertices to direct computation? (but then we lose the ldbc_id, if we remove it in Vertex class)
    // Lookups for entity and relation: (e.g. (0 -> knows), (1 -> isLocatedIn), ... )
    unordered_map<int, string> entityLookup;
    unordered_map<int, string> relationLookup;

    // Vertex data from tsv-files: unordered_map { global_id -> (entity.id, ldbc.id) }
    unordered_map<uint64_t, pair<int, uint64_t>> vertexDict;

    // Relationship data from tsv-files: vector of struct Relation (fromID, ToID, rel.id)
    vector<Relation> relationDict;

    // TODO: get base directory with cin -> user input
    string base = "/home/tim/Documents/TUD/(8) Informatik SS 2019/LDBC_Graph_Generating/LDBC_Python_Files/";

    importDataLookup(base + "relationLookup.tsv", relationLookup);
    importDataVertex(base + "entityDict.tsv", vertexDict, entityLookup); // entityLookup is built within the function automatically
    importDataRelations(base + "relationDict.tsv", relationDict);

    // --------------------------------------- Generating the graph ---------------------------------------

    morphstore::Graph ldbc_graph;
    generateVertices(vertexDict, ldbc_graph);
    generateEdges(relationDict, ldbc_graph);

    //ldbc_graph.printVertexByID(90563);
    ldbc_graph.statistics();

    return 0;
}