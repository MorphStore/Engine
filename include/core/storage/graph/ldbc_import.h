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
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>

namespace morphstore{

    class LDBC_Import{

    private:
        std::string directory;
        std::vector<std::string> verticesPaths;
        std::vector<std::string> relationsPaths;


    public:
        // constructor
        LDBC_Import(const std::string& dir){
            directory = dir;
            insert_file_names(directory);
        }

        std::string getDirectory() const{
            return directory;
        }

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

        // This function generates the the vertices from the vertex-vector
        void generate_Vertices(){
            // data structure for attributes: entity -> (attributes), e.g. tagclass -> (id, name, url)
            std::unordered_map<std::string, std::vector<std::string>> attributes;

            if(!verticesPaths.empty()) {
                std::cout << "Generating LDBC-Vertices ..." << std::endl;
                std::cout.flush();

                // (1) calculate global size to allocate
                for (const auto &address : verticesPaths) {

                    // get the entity from address ([...path...] / [entity-name].csv) and put key into attributes map
                    std::string entity = address.substr(getDirectory().size(), address.size() - getDirectory().size() - 4);
                    attributes[entity];

                    char* buffer;

                    uint64_t fileSize = 0;

                    std::ifstream vertexFile(address, std::ios::binary | std::ios::ate); // 'ate' means: open and seek to end immediately after opening

                    if (!vertexFile) {
                        std::cerr << "Error, opening file. ";
                        exit(EXIT_FAILURE);
                    }

                    if (vertexFile.is_open()) {
                        fileSize = vertexFile.tellg(); // tellg() returns: The current position of the get pointer in the stream on success, pos_type(-1) on failure.
                        vertexFile.clear();
                        vertexFile.seekg(0, std::ios::beg); // Seeks to the very beginning of the file, clearing any fail bits first (such as the end-of-file bit)
                    }

                    // (2) allocate memory
                    buffer = (char*) malloc( fileSize * sizeof( char ) );
                    vertexFile.read(buffer, fileSize); // read data as one big block
                    size_t start = 0;
                    std::string delimiter = "|";

                    // (3) do actual work with data in buffer ...
                    for(size_t i = 0; i < fileSize; ++i){
                        if(buffer[i] == '\n'){
                            // get a row into string form buffer with start- and end-point and do stuff ...
                            std::string row(&buffer[start], &buffer[i]);

                            // remove unnecessary '\n' at the beginning of a string
                            if(row.find('\n') != std::string::npos){
                                row.erase(0,1);
                            }

                            // handle first line of *.csv: contains the attributes; first attribute is ldbc_id -> important for edge-generation
                            if(row.rfind("id", 0) == 0){
                                // extract attribute from delimiter, e.g. id|name|url to id,name,url and push back to attributes vector
                                size_t last = 0;
                                size_t next = 0;
                                while ((next = row.find(delimiter, last)) != std::string::npos){
                                    attributes[entity].push_back(row.substr(last, next-last));
                                    last = next + 1;
                                }
                                // last attribute
                                attributes[entity].push_back(row.substr(last));
                            }else{
                                // (4) generate vertex with properties and write to graph
                            }

                            start = i; // set new starting point (otherwise it's concatenated)
                        }
                    }

                    delete[] buffer; // free memory
                    vertexFile.close();
                }

                // Verify
                for( std::unordered_map<std::string,std::vector<std::string> >::const_iterator ptr=attributes.begin(); ptr!=attributes.end(); ptr++) {
                    std::cout << ptr->first << ": ";
                    for( std::vector<std::string>::const_iterator eptr=ptr->second.begin(); eptr!=ptr->second.end(); eptr++){
                        std::cout << *eptr << " ";
                    }
                    std::cout << std::endl;
                }

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
