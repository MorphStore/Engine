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
 * @file cvertex.h
 * @brief Derived vertex calss for CSR storage format
 * @todo
*/

#ifndef MORPHSTORE_CVERTEX_H
#define MORPHSTORE_CVERTEX_H

namespace morphstore{

    class CSRVertex: public Vertex{

    public:
        // constructor with unique id generation
        CSRVertex(){
            // unique ID generation
            static uint64_t startID = 0;
            id = startID++;
        }

        // this function has no usage here: the adding of edges happens in the graph file -> csr.h
        // it's just here because it's a pure function in Vertex.h
        void add_edge(uint64_t from, uint64_t to,unsigned short int rel) override {
            std::cout << " virtual add_edge - no usage: " << from << ", " << to << ", " << rel << std::endl;
        }

        // pure function -> no functionality
        void add_edges(const std::vector<morphstore::Edge> edges) override {
            std::cout << " virtual add_edge - no usage: " << edges[0].getSourceId() << std::endl;
        }

        // debugging
        void print_neighbors() override {
            std::cout << " virtual print_neighbors - no usage: " << std::endl;
        }

        // get size of csr vertex in bytes:
        size_t get_data_size_of_vertex() override {
            size_t size = 0;
            // properties:
            size += sizeof(std::unordered_map<std::string, std::string>);
            for(std::unordered_map<std::string, std::string>::iterator property = properties.begin(); property != properties.end(); ++property){
                size += sizeof(char)*(property->first.length() + property->second.length());
            }
            // entity:
            size += sizeof(unsigned short int);
            // id
            size += sizeof(uint64_t);

            return size;
        }

    };
}

#endif //MORPHSTORE_CVERTEX_H
