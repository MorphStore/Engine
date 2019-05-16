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
 * @file vertex.h
 * @brief vertex class and its functions + Edge struct
 * @todo Add data structure for properties
*/

#ifndef MORPHSTORE_VERTEX_H
#define MORPHSTORE_VERTEX_H

#include <vector>
#include <iostream>
#include <unordered_map>


namespace morphstore{

    class Vertex;

    // this struct represents a relation to a target vertex; relation is the number in the lookup table
    struct Edge{
        Vertex* target;
        int relation;
    };

    class Vertex{

    private:
        // Vertex contains a (global) id; (old) ldbc id; entity number for lookup; vector adjList for the adjacency List
        uint64_t id;
        // TODO: remove ldbc_id from Vertex schema (to get more general structure without ldbc-dependency)
        uint64_t ldbc_id;
        std::vector<Edge> adjList;

        // properties
        std::unordered_map<std::string, std::string> properties;

    public:

        // constrcutor without the adjList (Vertex can contain no edges int the graph)
        Vertex(uint64_t id, uint64_t ldbc_id){
            SetVertex(id, ldbc_id);
        }

        void SetVertex(uint64_t id, uint64_t ldbc_id){
            this->id = id;
            this->ldbc_id = ldbc_id;
        }

        uint64_t getId() const{
            return id;
        }

        uint64_t getLDBC_Id(){
            return ldbc_id;
        }

        // returns a reference (read-only) of the adjacency list
        const std::vector<Edge>& getAdjList() const{
            return adjList;
        }

        void setProperties(std::unordered_map<std::string, std::string>& properties){
            if(!properties.empty()){
                this->properties = properties;
            }else{
                std::cout << "The properties-list is empty!" << std::endl;
            }
        }

        // function to add new neighbor vertex
        void add_edge(Vertex *target, int rel){
            Edge e;
            e.relation = rel;
            e.target = target;
            this->adjList.push_back(e);
        }

        int get_number_of_edges(){
            return adjList.size();
        }
    };
}

#endif //MORPHSTORE_VERTEX_H
