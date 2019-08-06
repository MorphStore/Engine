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
 * @file avertex.h
 * @brief Derived vertex calss for ADJ_LIST storage format
 * @todo change adjlist (vector of Edges) to vector of Edge* ?????
*/

#ifndef MORPHSTORE_AVERTEX_H
#define MORPHSTORE_AVERTEX_H

#include "../edge/edge.h"

namespace morphstore{

    class AVertex: public Vertex{

    protected:
        std::vector<Edge> adjList;

    public:
        // constructor with unique id generation
        AVertex(){
            // unique ID generation
            static uint64_t startID = 0;
            id = startID++;
        }

        // returns a reference (read-only) of the adjacency list
        const std::vector<Edge>& get_adjList() const{
            return adjList;
        }

        // add edge to vertexs' adjacencylist
        void add_edge(uint64_t from, uint64_t to, unsigned short int rel) override {
            Edge e(from, to, rel);
            this->adjList.push_back(e);
        }

        // function which returns the number of edges
        uint64_t get_number_edges() override {
            return adjList.size();
        }

        /* old-calculation of vertex size
        size_t get_size_of_vertex() {
            size_t size = 0;
            size += sizeof(uint64_t); // id
            // Adj.List:
            for(const auto& e : adjList){
                size += e.size_in_bytes();
            }
            // properties:
            size += sizeof(std::unordered_map<std::string, std::string>);
            for(std::unordered_map<std::string, std::string>::iterator property = properties.begin(); property != properties.end(); ++property){
                size += sizeof(char)*(property->first.length() + property->second.length());
            }
            // entities:
            size += sizeof(unsigned short int);

            return size;
        }
         */

    };
}

#endif //MORPHSTORE_AVERTEX_H
