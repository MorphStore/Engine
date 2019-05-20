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
 * @brief this class reads the ldbc files and generates the graph
 * @todo Any TODOS?
*/

#ifndef MORPHSTORE_LDBC_IMPORT_H
#define MORPHSTORE_LDBC_IMPORT_H

#include <experimental/filesystem>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
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

    class LDBC_Import{

    private:
        std::string directory;
        std::vector<std::string> verticesPaths;
        std::vector<std::string> relationsPaths;
        std::vector<std::string> entities;
        // data structure for lookup local ids with entity to global system id: (entity, ldbc_id) -> global id
        std::unordered_map< std::pair<std::string, std::string > , uint64_t , hash_pair> globalIdLookupMap;


    public:

        LDBC_Import(const std::string& dir){
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
        void generate_vertices(morphstore::Graph &graph){

            if(!verticesPaths.empty()) {
                std::cout << "(1/2) Generating LDBC-Vertices ...";
                std::cout.flush();

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
                                // add entity
                                properties.insert(std::make_pair("entity", entity));
                                //-----------------------------------------------------
                                // create vertex and insert into graph with properties
                                Vertex v;
                                graph.add_vertex_with_properties(v, properties);
                                // map entity and ldbc id to system generated id
                                globalIdLookupMap.insert({{entity, ldbcID}, v.getId()});
                                //-----------------------------------------------------
                                properties.clear(); // free memory
                            }

                            start = i; // set new starting point for buffer (otherwise it's concatenated)
                        }
                    }

                    delete[] buffer; // free memory
                    vertexFile.close();
                    // insert entity into vector
                    entities.push_back(entity);
                }
                std::cout << " --> done" << std::endl;
            }
        }

        // function which returns true, if parameter is a entity in ldbc-files
        bool isEntity(const std::string& entity){
            // iterate through entities vector to look up for paramater
            if (std::find(entities.begin(), entities.end(), entity) != entities.end()){
                return true;
            }
            return false;
        }


        // this function reads the relation-files and generates edges in graph
        void generate_edges(morphstore::Graph& graph){

            if(!relationsPaths.empty()) {
                std::cout << "(2/2) Generating LDBC-Edges ...";
                std::cout.flush();

                // iterate through vector of vertex-addresses
                for (const auto &address : relationsPaths) {

                    // get the relation-infos from file name: e.g. ([...path...] / [person_likes_comment].csv) --> person_likes_comment
                    std::string relation = address.substr(getDirectory().size(), address.size() - getDirectory().size() - 4);
                    std::string fromEntity = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string relationName = relation.substr(0, relation.find('_'));
                    relation.erase(0, relation.find('_') + 1);

                    std::string toEntity = relation;

                    // check from file name whether it's a relation file or multi value attribute file
                    // TODO: change handling of multi-value attributes (now just skipping...)
                    if(!isEntity(toEntity)){
                        // multiple attribute; toEntity in file-name is no entity -> e.g. isEntity("email") == false
                        std::cout << "\tFile is a multi-value attribute file. Skipping!" << std::endl;
                    }
                        // handling of relation-files ...
                    else{

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
                        bool hasProperties = false;
                        std::string propertyKey;

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
                                    uint64_t fromID = globalIdLookupMap.at({fromEntity, row.substr(0, row.find(delimiter))});
                                    // remove from id from string
                                    row.erase(0, row.find(delimiter) + delimiter.length());
                                    std::string value;
                                    uint64_t toID;
                                    if(!hasProperties){
                                        // WITHOUT properties: just from the first delimiter on
                                        toID = globalIdLookupMap.at({toEntity, row});

                                        // Generate edge in graph
                                        graph.add_edge(fromID, toID, relationName);
                                    }else{
                                        // with properties means: toID is until the next delimiter, and then the value for the property
                                        toID = globalIdLookupMap.at({toEntity, row.substr(0, row.find(delimiter))});
                                        row.erase(0, row.find(delimiter) + delimiter.length());
                                        value = row;
                                        graph.add_edge_with_property(fromID, toID, relationName, {propertyKey, value});
                                    }
                                }
                                start = i; // set new starting point for buffer (otherwise it's concatenated)
                            }
                        }
                        delete[] buffer; // free memory
                        relationFile.close();
                    }
                }
                globalIdLookupMap.clear(); // we dont need the lookup anymore -> delete memory
                std::cout << " --> done" << std::endl;
            }
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

    };
}

#endif //MORPHSTORE_LDBC_IMPORT_H
