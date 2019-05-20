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
 * @file graph.h
 * @brief Graph storage format -> adjacency Lists
 * @todo
 */

#ifndef MORPHSTORE_GRAPH_H
#define MORPHSTORE_GRAPH_H

#include <core/storage/graph/vertex.h>

#include <unordered_map>
#include <vector>
#include <iostream>


namespace morphstore{

    class Graph{

    private:
        // main data structure: mapping global id -> vertex
        std::unordered_map<uint64_t, Vertex> vertices;

    public:

        // function to add a new (ldbc) vertex to the graph
        void add_vertex(const Vertex& v){
            if(!exist_id(v.getId())){
                Vertex v;
                vertices.insert(std::make_pair(v.getId(), v));
            }else{
                std::cout << "Vertex with ID " << v.getId() << " already exists in the database!";
            }
        }

        // function to add a new (ldbc) vertex to the graph
        void add_vertex_with_properties(Vertex& v, std::unordered_map<std::string, std::string>& props ){
            if(!exist_id(v.getId())){
                v.set_properties(props);
                vertices.insert(std::make_pair(v.getId(), v));
            } else{
                std::cout << "Vertex with ID " << v.getId() << " already exists in the database!";
            }
        }

        // function that creates a new relation/edge between two (existing) vertices
        void add_edge(uint64_t sourceID, uint64_t targetID, std::string rel){
            if(exist_id(sourceID) && exist_id(targetID)){
                Vertex* sourceV = &vertices.at(sourceID);
                Vertex* targetV = &vertices.at(targetID);
                sourceV->add_edge(targetV, rel);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!";
            }
        }

        // function that creates a new relation/edge between two (existing) vertices WITH property
        void add_edge_with_property(uint64_t sourceID, uint64_t targetID, std::string rel, std::pair<std::string, std::string> property){
            if(exist_id(sourceID) && exist_id(targetID)){
                Vertex* sourceV = &vertices.at(sourceID);
                Vertex* targetV = &vertices.at(targetID);
                sourceV->add_edge_with_property(targetV, rel, property);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!";
            }
        }

        // function to check if the ID is present or not
        bool exist_id(const uint64_t id){
            if(vertices.find(id) == vertices.end()){
                return false;
            }
            return true;
        }

        // this function returns the total number of edges in the graph
        int get_total_number_of_edges(){
            uint64_t totalNumberEdges = 0;
            for(std::unordered_map<uint64_t, Vertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                totalNumberEdges += it->second.get_number_of_edges();
            }
            return static_cast<int>(totalNumberEdges);
        }

        // for debbuging
        void statistics(){
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << vertices.size() << std::endl;
            std::cout << "Number of relations/edges: " << get_total_number_of_edges() << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
        }

        // for debugging
        void print_vertex_by_id(uint64_t id){
            std::cout << "-------------- Vertex ID: " << id <<" --------------" << std::endl;
            Vertex* v = &vertices.at(id);
            std::cout << "Vertex-ID: \t"<< v->getId() << std::endl;
            std::cout << "#Edges: \t" << v->get_adjList().size() << std::endl;
            std::cout << "Adj.List: ";

            const std::vector<Edge>& adjList = v->get_adjList();
            for(const auto& e : adjList){
                std::cout << "(" << e.target->getId() << "," << e.relation << ") ";
            }
            std::cout << "\n";
            std::cout << "Properties: "; v->print_properties();
            std::cout << "\n";
        }
    };

}

#endif //MORPHSTORE_GRAPH_H
