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

        // intermediate vertex data structure: (entity, ldbc_id) -> properties [key is pair because we have to handle the local ids -> identification]
        std::unordered_map< std::pair<std::string, std::string > , std::unordered_map<std::string, std::string>, hash_pair> verticesMap;


    public:

        // constructor
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

        // this function reads the vertices-files and write it to the intermediate map verticesMap
        void read_data_vertices(){

            if(!verticesPaths.empty()) {
                std::cout << "Reading LDBC-Vertices ...";
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
                            if(row.rfind("id", 0) == 0){
                                // extract attribute from delimiter, e.g. id|name|url to id,name,url and push back to attributes vector
                                while ((next = row.find(delimiter, last)) != std::string::npos){
                                    attributes.push_back(row.substr(last, next-last));
                                    last = next + 1;
                                }
                                // last attribute
                                attributes.push_back(row.substr(last));
                            }else{
                                // actual data: write to intermediate properties map
                                std::unordered_map<std::string, std::string> properties;
                                size_t attrIndex = 0;
                                while ((next = row.find(delimiter, last)) != std::string::npos){
                                    properties.insert(std::make_pair(attributes[attrIndex], row.substr(last, next-last)));
                                    last = next + 1;
                                    ++attrIndex;
                                }
                                // last attribute
                                properties.insert(std::make_pair(attributes[attrIndex], row.substr(last)));
                                // add entity
                                properties.insert(std::make_pair("entity", entity));
                                //insert into main importer data structure
                                verticesMap.insert({{entity, row.substr(0, row.find(delimiter))}, properties});
                                properties.clear(); // free memory
                            }

                            start = i; // set new starting point for buffer (otherwise it's concatenated)
                        }
                    }

                    delete[] buffer; // free memory
                    vertexFile.close();
                }
                std::cout << " --> done" << std::endl;
            }
        }

        // function which generates the vertices to a given graph
        void generate_vertices_in_graph(Graph& graph){
            // for every vertex in the intermediate verticesMap, get properties map and insert into graph
            for(const auto& vertex : verticesMap){
                std::unordered_map<std::string, std::string> props = verticesMap.at(vertex.first);
                Vertex v;
                graph.add_vertex_with_properties(v, props);
            }
            // clear vector
            verticesMap.clear();
        }

        // this function reads the relation-files and write it to the intermediate map verticesMap
        void read_data_edges(){

            if(!verticesPaths.empty()) {
                std::cout << "Generating LDBC-Edges ...";
                std::cout.flush();

                // iterate through vector of vertex-addresses
                for (const auto &address : relationsPaths) {

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

                            start = i; // set new starting point for buffer (otherwise it's concatenated)
                        }
                    }

                    delete[] buffer; // free memory
                    vertexFile.close();
                }
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

        // debugging
        void print_vertex_at(std::string entity, std::string ldbc_id){
            std::pair<std::string, std::string> key = {entity, ldbc_id};
            std::unordered_map<std::string, std::string> searchedObject = verticesMap.at(key);
            std::cout << "Vertex={ ldbc_id=" << ldbc_id << " ";
            for(const auto& attr : searchedObject) {
                std::cout << attr.first << "=" << attr.second << " ";
            }
            std::cout << " }" << std::endl;
        }

    };



}

#endif //MORPHSTORE_LDBC_IMPORT_H
