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
 * @file ldbc_import.h
 * @brief this class reads the ldbc files and generates the graph in CSR or AdjList
 * @todo CSR-EDGE PROPERTIES ARE MISSING!!!
*/

#ifndef MORPHSTORE_LDBC_IMPORT_H
#define MORPHSTORE_LDBC_IMPORT_H

#include <core/storage/graph/adj_list/graph.h>
#include <core/storage/graph/csr/graph.h>

#include <experimental/filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <map>
#include <algorithm>

// hash function used to hash a pair of any kind using XOR (for verticesMap)
struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ hash2;
    }
};

namespace morphstore{

    class LDBCImport{

    private:
        std::string directory;
        std::vector<std::string> verticesPaths;
        std::vector<std::string> relationsPaths;
        std::map<unsigned short int, std::string> entitiesLookup;
        std::map<unsigned short int, std::string> relationsLookup;
        // data structure for lookup local ids with entity to global system id: (entity, ldbc_id) -> global id
        std::unordered_map< std::pair<std::string, std::string > , uint64_t , hash_pair> globalIdLookupMap;

        // for CSR data structure
        // map for lookup every system-id, the neighbors in the graph (for further processing, e.g. filling the edge_array in the right order)
        std::unordered_map< uint64_t, std::vector<std::pair<uint64_t , unsigned short int >>> vertexNeighborsLookup;

    public:

        LDBCImport(const std::string& dir){
            directory = dir;
            insert_file_names(directory);
        }

        std::string getDirectory() const{
            return directory;
        }

        // function which iterates through directory to receive file names (entire path)
        void insert_file_names(std::string dir){
            for (const auto & entry : std::experimental::filesystem::directory_iterator(dir)){
                // ignore files starting with a '.'
                if(entry.path().string()[dir.size()] == '.'){
                    continue;
                }else{
                    // insert file path to vertices or relations vector
                    differentiate(entry.path().string(), dir);
                }
            }
        }

        // this function differentiates, whether the file is a vertex or relation and puts it into the specific vector
        void differentiate(std::string path, std::string dir){
            // if the string contains a '_' -> it's a relation file; otherwise a vertex file
            // remove dir name to remain only the *.csv
            if(path.substr(dir.size()).find('_') != std::string::npos ){
                relationsPaths.push_back(path);
            }else{
                verticesPaths.push_back(path);
            }
        }

        // this function reads the vertices-files and creates vertices in a graph
        template <class T>
        void generate_vertices(T &graph){

            if(!verticesPaths.empty()) {
                //std::cout << "(1/2) Generating LDBC-Vertices ...";
                //std::cout.flush();

                //this variable is used for the entityLookup-keys, starting by 0
                unsigned short int entityNumber = 0;

                // iterate through vector of vertex-addresses
                for (const auto &address : verticesPaths) {

                    // data structure for attributes of entity, e.g. taglass -> id, name, url
                    std::vector<std::string> attributes;

                    // get the entity from address ([...path...] / [entity-name].csv)
                    std::string entity = address.substr(getDirectory().size(), address.size() - getDirectory().size() - 4);

                    char* buffer;

                    uint64_t fileSize = 0;

                    std::ifstream vertexFile(address, std::ios::binary | std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!vertexFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    // calculate file size
                    if (vertexFile.is_open()) {
                        fileSize = static_cast<uint64_t>(vertexFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        vertexFile.clear();
                        vertexFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // allocate memory
                    buffer = (char*) malloc( fileSize * sizeof( char ) );
                    vertexFile.read(buffer, fileSize); // read data as one big block
                    size_t start = 0;
                    std::string delimiter = "|";

                    // read buffer and do the magic ...
                    for(size_t i = 0; i < fileSize; ++i){
                        if(buffer[i] == '\n'){
                            // get a row into string form buffer with start- and end-point
                            std::string row(&buffer[start], &buffer[i]);

                            // remove unnecessary '\n' at the beginning of a string
                            if(row.find('\n') != std::string::npos){
                                row.erase(0,1);
                            }

                            size_t last = 0;
                            size_t next = 0;

                            // first line of *.csv contains the attributes -> write to attributes vector
                            if(start == 0){
                                // extract attribute from delimiter, e.g. id|name|url to id,name,url and push back to attributes vector
                                while ((next = row.find(delimiter, last)) != std::string::npos){
                                    attributes.push_back(row.substr(last, next-last));
                                    last = next + 1;
                                }
                                // last attribute
                                attributes.push_back(row.substr(last));
                            }else{
                                // actual data:
                                std::unordered_map<std::string, std::string> properties;
                                size_t attrIndex = 0;
                                std::string ldbcID = row.substr(0, row.find(delimiter));
                                while ((next = row.find(delimiter, last)) != std::string::npos){
                                    properties.insert(std::make_pair(attributes[attrIndex], row.substr(last, next-last)));
                                    last = next + 1;
                                    ++attrIndex;
                                }
                                // last attribute
                                properties.insert(std::make_pair(attributes[attrIndex], row.substr(last)));

                                //-----------------------------------------------------
                                // create vertex and insert into graph with properties
                                uint64_t systemID = graph.add_vertex_with_properties(properties);
                                // add entity number to vertex
                                graph.add_entity_to_vertex(systemID, entityNumber);
                                // map entity and ldbc id to system generated id
                                globalIdLookupMap.insert({{entity, ldbcID}, systemID});
                                //-----------------------------------------------------
                                properties.clear(); // free memory
                            }

                            start = i; // set new starting point for buffer (otherwise it's concatenated)
                        }
                    }

                    delete[] buffer; // free memory
                    vertexFile.close();

                    // insert entity-number with string into map
                    entitiesLookup.insert(std::make_pair( entityNumber, entity));
                    ++entityNumber;
                }
                // graph gets full entity-list here:
                graph.set_entity_dictionary(entitiesLookup);
            }

        }

        // function which returns true, if parameter is a entity in ldbc-files
        bool is_entity(const std::string &entity){
            // iterate through entities-map to look up for paramater
            for(auto const& entry : entitiesLookup){
                if(entry.second == entity){
                    return true;
                }
            }

            return false;
        }

        // function which returns true, if the relation already exist
        bool exist_relation_name(const std::string& relation){
            // iterate through relations-map to look up for paramater
            for(auto const& entry : relationsLookup){
                if(entry.second == relation){
                    return true;
                }
            }

            return false;
        }

        // for debugging
        void print_file_names(){
            std::cout << "Vertices-Files: " << std::endl;
            for(const auto& v : verticesPaths){
                std::cout << "\t" << v << std::endl;
            }
            std::cout << "Relations-Files: " << std::endl;
            for(const auto& rel : relationsPaths){
                std::cout << "\t" << rel << std::endl;
            }

        }

        // function which clears all intermediates after import
        void clear_intermediates(){
            globalIdLookupMap.clear();
            relationsLookup.clear();
            entitiesLookup.clear();
            relationsPaths.clear();
            verticesPaths.clear();
        }

        // function which returns the total number of edges (IMPORTANT: vertex generation has to be done first, because of the entity lookup creation)
        uint64_t get_total_number_edges(){

            uint64_t result = 0 ;

            if(!relationsPaths.empty()) {

                // iterate through vector of relation-addresses
                for (const auto &address : relationsPaths) {

                    // TODO OPTIMIZE HERE: remove string operations
                    // get the relation-infos from file name: e.g. ([...path...] / [person_likes_comment].csv) --> person_likes_comment
                    std::string relation = address.substr(getDirectory().size(), address.size() - getDirectory().size() - 4);
                    std::string fromEntity = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string relationName = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string toEntity = relation;

                    char* buffer;

                    uint64_t fileSize = 0;

                    std::ifstream relationFile(address, std::ios::binary | std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!relationFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    // calculate file size
                    if (relationFile.is_open()) {
                        fileSize = static_cast<uint64_t>(relationFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        relationFile.clear();
                        relationFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // allocate memory
                    buffer = (char*) malloc( fileSize * sizeof( char ) );
                    relationFile.read(buffer, fileSize); // read data as one big block
                    bool firstLine = true;

                    // check from file name whether it's a relation file or multi value attribute file
                    if(is_entity(toEntity)){

                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // skip first line (attributes infos....)
                                if(firstLine){
                                    firstLine = false;
                                }else{
                                    ++result;
                                }
                            }
                        }

                    }

                    delete[] buffer; // free memory
                    relationFile.close();

                }
            }
            return result;
        }



        // -------------------------------- Adj-List-specific functions --------------------------------

        // Import into Adj-List-Format:
        // generate_vertices() + generate_edges()
        void import(morphstore::AdjacencyList &graph){
            std::cout << "Importing LDBC-files into graph ... ";
            std::cout.flush();

            // (1) generate vertices
            generate_vertices(graph);
            // (2) generate edges
            generate_edges_adj_list(graph);

            // (3) clear intermediates
            clear_intermediates();

            std::cout << "--> done" << std::endl;
        }

        // this function reads the relation-files and generates edges in graph
        void generate_edges_adj_list(morphstore::AdjacencyList &graph){



            if(!relationsPaths.empty()) {
                //std::cout << "(2/2) Generating LDBC-Edges ...";
                //std::cout.flush();

                //this variable is used for the relationLookup-keys, starting by 0
                unsigned short int relationNumber = 0;
                bool isRelation = false; // flag which is used to differentiate for relatoin-lookup-entrys (to avoid e.g. email as relation)

                // iterate through vector of vertex-addresses
                for (const auto &address : relationsPaths) {

                    isRelation = false;

                    // get the relation-infos from file name: e.g. ([...path...] / [person_likes_comment].csv) --> person_likes_comment
                    std::string relation = address.substr(getDirectory().size(), address.size() - getDirectory().size() - 4);
                    std::string fromEntity = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string relationName = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string toEntity = relation;

                    char* buffer;

                    uint64_t fileSize = 0;

                    std::ifstream relationFile(address, std::ios::binary | std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!relationFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    // calculate file size
                    if (relationFile.is_open()) {
                        fileSize = static_cast<uint64_t>(relationFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        relationFile.clear();
                        relationFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // allocate memory
                    buffer = (char*) malloc( fileSize * sizeof( char ) );
                    relationFile.read(buffer, fileSize); // read data as one big block

                    size_t start = 0;
                    std::string delimiter = "|";

                    // check from file name whether it's a relation file or multi value attribute file
                    if(!is_entity(toEntity)){
                        // Multi-value-attributes: just take the last recently one
                        std::string propertyKey;
                        std::unordered_map<uint64_t, std::string> multiValueAttr;
                        uint64_t systemID;
                        std::string value;

                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // get a row into string form buffer with start- and end-point
                                std::string row(&buffer[start], &buffer[i]);

                                // remove unnecessary '\n' at the beginning of a string
                                if(row.find('\n') != std::string::npos){
                                    row.erase(0,1);
                                }

                                // first line: get the attribute a.k.a key for the property, e.g. Person.id|email -> get 'email'
                                if(start == 0){
                                    propertyKey = row.substr(row.find(delimiter) + 1);
                                }else{
                                    // (1) write data to vector: if key is already present, over write value (simplicity: we take the newest one)
                                    systemID = globalIdLookupMap.at({fromEntity, row.substr(0, row.find(delimiter))});
                                    value = row.substr(row.find(delimiter) + 1);
                                    multiValueAttr[systemID] = std::move(value);
                                }

                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                        // iterate through multiValue map and assign property to vertex
                        for(const auto &pair : multiValueAttr){
                            const std::pair<std::string, std::string>& keyValuePair = {propertyKey, pair.second};
                            graph.add_property_to_vertex(pair.first, keyValuePair);
                        }

                    }
                        // handling of relation-files ...
                    else{

                        isRelation = true;

                        bool hasProperties = false;
                        std::string propertyKey;
                        uint64_t fromID, toID;

                        // read buffer and do the magic ...
                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // get a row into string form buffer with start- and end-point
                                std::string row(&buffer[start], &buffer[i]);

                                // remove unnecessary '\n' at the beginning of a string
                                if(row.find('\n') != std::string::npos){
                                    row.erase(0,1);
                                }

                                size_t last = 0;
                                size_t next = 0;
                                size_t count = 0;

                                // first line of *.csv: Differentiate whether it's
                                // (1) relation without properties: e.g. Person.id|Person.id -> #delimiter = 1
                                // (2) relation with properties: e.g. Person.id|Person.id|fromDate -> #delimiter = 2
                                if(start == 0){
                                    // if there are 2 delimiter ('|') -> relation file with properties
                                    while ((next = row.find(delimiter, last)) != std::string::npos){
                                        last = next + 1;
                                        ++count;
                                    }
                                    if(count == 2){
                                        hasProperties = true;
                                        propertyKey = row.substr(last);
                                    }
                                }else{
                                    // lines of data: (from_local-ldbc-id), (to_local-ldbc-id) and property
                                    // get the system-(global) id's from local ids
                                    fromID = globalIdLookupMap.at({fromEntity, row.substr(0, row.find(delimiter))});
                                    // remove from id from string
                                    row.erase(0, row.find(delimiter) + delimiter.length());
                                    std::string value;
                                    if(!hasProperties){
                                        // WITHOUT properties: just from the first delimiter on
                                        toID = globalIdLookupMap.at({toEntity, row});

                                        // Generate edge in graph
                                        graph.add_edge(fromID, toID, relationNumber);
                                    }else{
                                        // with properties means: toID is until the next delimiter, and then the value for the property
                                        toID = globalIdLookupMap.at({toEntity, row.substr(0, row.find(delimiter))});
                                        row.erase(0, row.find(delimiter) + delimiter.length());
                                        value = row;
                                        graph.add_edge_with_property(fromID, toID, relationNumber, {propertyKey, value});
                                    }
                                }
                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                    }
                    delete[] buffer; // free memory
                    relationFile.close();

                    //check if the relation name is a relation (no multi value file)
                    if(isRelation){
                        // check if the name already exists
                        if(!exist_relation_name(relationName)){
                            // insert relation-number with string into map
                            relationsLookup.insert(std::make_pair( relationNumber, relationName));
                            ++relationNumber;
                        }
                    }

                }
                // graph gets full relation-list here:
                graph.set_relation_dictionary(relationsLookup);
            }
        }



        // -------------------------------- CSR-specific functions --------------------------------

        // Import into CSR-Format:
        // generate_vertices() + generate_edges()
        void import(morphstore::CSR &graph){
            std::cout << "Importing LDBC-files into graph ... ";
            std::cout.flush();

            // (1) generate vertices
            generate_vertices(graph);
            // (2) allocate memory
            allocate_graph_structure_memory_csr(graph);
            // (3) generate edges
            generate_edges_csr(graph);

            // (4) remove intermediates
            clear_intermediates();

            std::cout << "--> done" << std::endl;
        }

        // this function reads the relation-files and generates edges in graph
        void generate_edges_csr(morphstore::CSR &graph){



            if(!relationsPaths.empty()) {
                //std::cout << "(2/2) Generating LDBC-Edges ...";
                //std::cout.flush();

                //this variable is used for the relationLookup-keys, starting by 0
                unsigned short int relationNumber = 0;
                bool isRelation = false; // flag which is used to differentiate for relatoin-lookup-entrys (to avoid e.g. email as relation)

                // iterate through vector of vertex-addresses
                for (const auto &address : relationsPaths) {

                    isRelation = false;

                    // get the relation-infos from file name: e.g. ([...path...] / [person_likes_comment].csv) --> person_likes_comment
                    std::string relation = address.substr(getDirectory().size(), address.size() - getDirectory().size() - 4);
                    std::string fromEntity = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string relationName = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string toEntity = relation;

                    char* buffer;

                    uint64_t fileSize = 0;

                    std::ifstream relationFile(address, std::ios::binary | std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!relationFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    // calculate file size
                    if (relationFile.is_open()) {
                        fileSize = static_cast<uint64_t>(relationFile.tellg()); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        relationFile.clear();
                        relationFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // allocate memory
                    buffer = (char*) malloc( fileSize * sizeof( char ) );
                    relationFile.read(buffer, fileSize); // read data as one big block

                    size_t start = 0;
                    std::string delimiter = "|";

                    // check from file name whether it's a relation file or multi value attribute file
                    if(!is_entity(toEntity)){
                        // Multi-value-attributes: just take the last recently one
                        std::string propertyKey;
                        std::unordered_map<uint64_t, std::string> multiValueAttr;
                        uint64_t systemID;
                        std::string value;

                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // get a row into string form buffer with start- and end-point
                                std::string row(&buffer[start], &buffer[i]);

                                // remove unnecessary '\n' at the beginning of a string
                                if(row.find('\n') != std::string::npos){
                                    row.erase(0,1);
                                }

                                // first line: get the attribute a.k.a key for the property, e.g. Person.id|email -> get 'email'
                                if(start == 0){
                                    propertyKey = row.substr(row.find(delimiter) + 1);
                                }else{
                                    // (1) write data to vector: if key is already present, over write value (simplicity: we take the newest one)
                                    systemID = globalIdLookupMap.at({fromEntity, row.substr(0, row.find(delimiter))});
                                    value = row.substr(row.find(delimiter) + 1);
                                    multiValueAttr[systemID] = std::move(value);
                                }

                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                        // iterate through multiValue map and assign property to vertex
                        for(const auto &pair : multiValueAttr){
                            const std::pair<std::string, std::string>& keyValuePair = {propertyKey, pair.second};
                            graph.add_property_to_vertex(pair.first, keyValuePair);
                        }

                    }
                        // handling of relation-files ...
                    else{

                        isRelation = true;

                        bool hasProperties = false;
                        std::string propertyKey;
                        uint64_t fromID, toID;

                        // read buffer and do the magic ...
                        for(size_t i = 0; i < fileSize; ++i){
                            if(buffer[i] == '\n'){
                                // get a row into string form buffer with start- and end-point
                                std::string row(&buffer[start], &buffer[i]);

                                // remove unnecessary '\n' at the beginning of a string
                                if(row.find('\n') != std::string::npos){
                                    row.erase(0,1);
                                }

                                size_t last = 0;
                                size_t next = 0;
                                size_t count = 0;

                                // first line of *.csv: Differentiate whether it's
                                // (1) relation without properties: e.g. Person.id|Person.id -> #delimiter = 1
                                // (2) relation with properties: e.g. Person.id|Person.id|fromDate -> #delimiter = 2
                                if(start == 0){
                                    // if there are 2 delimiter ('|') -> relation file with properties
                                    while ((next = row.find(delimiter, last)) != std::string::npos){
                                        last = next + 1;
                                        ++count;
                                    }
                                    if(count == 2){
                                        hasProperties = true;
                                        propertyKey = row.substr(last);
                                    }
                                }else{
                                    // lines of data: (from_local-ldbc-id), (to_local-ldbc-id) and property
                                    // get the system-(global) id's from local ids
                                    fromID = globalIdLookupMap.at({fromEntity, row.substr(0, row.find(delimiter))});
                                    // remove from id from string
                                    row.erase(0, row.find(delimiter) + delimiter.length());
                                    std::string value;
                                    if(!hasProperties){
                                        // WITHOUT properties: just from the first delimiter on
                                        toID = globalIdLookupMap.at({toEntity, row});

                                        // Generate edge in graph
                                        //graph.add_edge(fromID, toID, relationNumber);

                                        // insert relation into vertexNeighborsLookup
                                        vertexNeighborsLookup[fromID].push_back({toID, relationNumber});
                                    }else{
                                        // with properties means: toID is until the next delimiter, and then the value for the property
                                        toID = globalIdLookupMap.at({toEntity, row.substr(0, row.find(delimiter))});
                                        row.erase(0, row.find(delimiter) + delimiter.length());
                                        value = row;
                                        // add to graph
                                        //graph.add_edge_with_property(fromID, toID, relationNumber, {propertyKey, value});
                                        vertexNeighborsLookup[fromID].push_back({toID, relationNumber});
                                    }
                                }
                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                    }
                    delete[] buffer; // free memory
                    relationFile.close();

                    //check if the relation name is a relation (no multi value file)
                    if(isRelation){
                        // check if the name already exists
                        if(!exist_relation_name(relationName)){
                            // insert relation-number with string into map
                            relationsLookup.insert(std::make_pair( relationNumber, relationName));
                            ++relationNumber;
                        }
                    }

                }
                // graph gets full relation-list here:
                graph.set_relation_dictionary(relationsLookup);

                // do actual edge generation here:
                write_intermediates_into_graph_csr(graph);
            }
        }

        // this function allocates the memory used for the graph structure in CSR (arrays)
        void allocate_graph_structure_memory_csr(morphstore::CSR &graph){
            // get number of vertices and number of edges
            uint64_t numberVertices = graph.getNumberVertices();
            uint64_t numberEdges = get_total_number_edges();
            graph.allocate_graph_structure_memory(numberVertices, numberEdges);
        }

        // function for sorting the vertexNeighborsLookup ASC in CSR
        void sort_VertexNeighborsLookup_csr(){
            // sorting the first element of the pair (target-id)
            for(auto &it: vertexNeighborsLookup){
                std::sort(it.second.begin(), it.second.end());
            }
        }

        // this function writes the actual data from the intermediate vertexNeighborsLookup int to the arrays in the csr format
        void write_intermediates_into_graph_csr(morphstore::CSR &graph){
            // firstly, sorting the intermediates with their target IDs ASC
            sort_VertexNeighborsLookup_csr();

            // Write CSR arrays with data (offsets, number of relation,....):
            uint64_t lastVertexID = graph.getNumberVertices() - 1;
            uint64_t startOffset = 0;

            for(uint64_t vertexID = 0; vertexID < lastVertexID; ++vertexID){
                // get the list of target vertices
                std::vector<std::pair<uint64_t , unsigned short int >> neighbors;
                neighbors = vertexNeighborsLookup[vertexID];
                //store the number for the offset in edge array
                uint64_t endOffset = neighbors.size() + startOffset -1 ;
                // VERTICES WITHOUT ANY EDGES -< TODO ? how to handle?
                graph.add_edge_ldbc(vertexID, startOffset, neighbors);

                startOffset = endOffset + 1 ;
            }
        }
    };
}

#endif //MORPHSTORE_LDBC_IMPORT_H
