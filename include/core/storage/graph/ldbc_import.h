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

namespace morphstore{

    class LDBC_Import{

    private:
        std::string directory;
        std::vector<std::string> verticesPaths;
        std::vector<std::string> relationsPaths;


    public:
        // constructor
        LDBC_Import(std::string dir){
            directory = dir;
            insert_file_names(dir);
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

        // for debugging
        void print_file_names(){
            std::cout << "Vertices-Files: " << std::endl;
            for(auto& v : verticesPaths){
                std::cout << "\t" << v << std::endl;
            }

            std::cout << "Relations-Files: " << std::endl;
            for(auto& rel : relationsPaths){
                std::cout << "\t" << rel << std::endl;
            }

        }



    };



}

#endif //MORPHSTORE_LDBC_IMPORT_H
