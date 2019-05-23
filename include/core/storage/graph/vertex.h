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
 * @todo
*/

#ifndef MORPHSTORE_VERTEX_H
#define MORPHSTORE_VERTEX_H

#include <vector>
#include <iostream>
#include <unordered_map>
#include <unordered_set>


namespace morphstore{

    class Vertex;

    // this struct represents a relation to a target vertex; relation is the number in the lookup table
    struct Edge{
        Vertex* target;
        std::string relation;
        std::pair<std::string, std::string> property;
    };

    class Vertex{

    private:
        // Vertex contains a (global) id; (old) ldbc id; entity number for lookup; vector adjList for the adjacency List
        uint64_t id;
        std::vector<Edge> adjList;
        // properties
        std::unordered_map<std::string, std::string> properties;
        // a vertex can have multiple entites
        std::unordered_set<std::string> entities;

    public:

        // constrcutor without the adjList (Vertex can contain no edges int the graph)
        Vertex(){
            // unique ID generation
            static uint64_t startID = 0;
            id = startID++;
        }

        uint64_t getId() const{
            return id;
        }

        // returns a reference (read-only) of the adjacency list
        const std::vector<Edge>& get_adjList() const{
            return adjList;
        }

        // this function adds a whole property map to a vertex
        void add_properties(std::unordered_map<std::string, std::string> &properties){
            if(!properties.empty()){
                this->properties = properties;
            }else{
                std::cout << "The properties-list is empty!" << std::endl;
            }
        }

        // this adds one key-value pair to the vertex's property map
        void add_property(const std::pair<std::string, std::string>& property){
            this->properties[property.first] = property.second;
        }

        // function that creates a new relation/edge between two (existing) vertices withouht properties
        void add_edge(Vertex *target, std::string relation){
            Edge e;
            e.target = target;
            e.relation = relation;
            this->adjList.push_back(e);
        }

        // add edge with properties to vertex
        void add_edge_with_property(Vertex *target, std::string relation, std::pair<std::string, std::string> property){
            Edge e;
            e.target = target;
            e.relation = relation;
            e.property = property;
            this->adjList.push_back(e);
        }

        // add entity to vertex
        void add_entity(std::string e){
            this->entities.insert(e);
        }

        int get_number_of_edges(){
            return static_cast<int>(adjList.size());
        }

        void print_properties(){
            for(const auto& entry : properties){
                std::cout << "{" << entry.first << ": " << entry.second << "}";
            }
        }
    };
}

#endif //MORPHSTORE_VERTEX_H
