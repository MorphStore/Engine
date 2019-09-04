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
 * @brief Derived vertex calss for ADJ_LIST storage format: base-class: vertex
 * @todo
*/

#ifndef MORPHSTORE_AVERTEX_H
#define MORPHSTORE_AVERTEX_H

#include "../edge/edge.h"

namespace morphstore{

    class AdjacencyListVertex: public Vertex{

    protected:
        std::vector<Edge> adjacencylist;

    public:
        // constructor with unique id generation
        AdjacencyListVertex(){
            // unique ID generation
            static uint64_t startID = 0;
            id = startID++;
        }

        // returns a reference (read-only) of the adjacency list
        const std::vector<Edge>& get_adjList() const{
            return adjacencylist;
        }

        // function to add a single edge to vertexs adjlist
        void add_edge(uint64_t from, uint64_t to, unsigned short int rel) override {
            this->adjacencylist.push_back(Edge(from, to, rel));
        }

        // add edges to vertexs' adjacencylist
        void add_edges(std::vector<morphstore::Edge> edges) override {
            /*
            adjacencylist.resize(edges.size());
            for(uint64_t i = 0; i < edges.size(); ++i){
                adjacencylist.push_back(edges[i]);
            }
             */

            this->adjacencylist = edges;
        }

        // function which returns the number of edges
        uint64_t get_number_edges() override {
            return adjacencylist.size();
        }

        void print_neighbors() override {
            for(const auto& edge : adjacencylist){
                std::cout << "Source-ID: " << edge.getSourceId() << " - Target-ID: " << edge.getTargetId() <<
                " - Property: { " << edge.getProperty().first << ": " << edge.getProperty().second << " }" << " || ";
            }
        }

        // function to return a vector of neighbor ids (for BFS)
        std::vector<uint64_t> get_neighbors_ids() override {
            std::vector<uint64_t> neighbors;
            for(auto const& edge : adjacencylist){
                neighbors.push_back(edge.getTargetId());
            }
            return neighbors;
        }

        size_t get_size_of_vertex() override {
            size_t size = 0;
            size += sizeof(uint64_t); // id
            size += sizeof(unsigned short int); // entity

            // properties:
            size += sizeof(std::unordered_map<std::string, std::string>);
            for(std::unordered_map<std::string, std::string>::iterator property = properties.begin(); property != properties.end(); ++property){
                size += sizeof(char)*(property->first.length() + property->second.length());
            }

            // Adj.List:
            size += sizeof(std::vector<Edge>);
            for(const auto& e : adjacencylist){
                size += e.size_in_bytes();
            }
            return size;
        }

    };
}

#endif //MORPHSTORE_AVERTEX_H
