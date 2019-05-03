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
 * @todo Add property structure to Vertex and Edge
 */

#ifndef MORPHSTORE_GRAPH_H
#define MORPHSTORE_GRAPH_H

#include <core/storage/vertex.h>

#include <unordered_map>
#include <vector>
#include <iostream>


namespace graph{

    class Graph{

    private:
        // main data structure: mapping global id -> vertex
        std::unordered_map<unsigned long int, Vertex> vertices;

    public:

        // function to add a new (ldbc) vertex to the graph
        void addVertex(unsigned long int id, unsigned long int ldbc_id, int entity){
            // if key is not present -> create vertex
            if(existID(id)){
                Vertex v(id, ldbc_id, entity);
                vertices.insert(std::make_pair(id, v));
            }else{
                std::cout << "Vertex with ID " << id << " already exists!";
            }
        }

        // function that creates a new relation/edge between two (existing) vertices
        void addEdge(unsigned long int sourceID, unsigned long int targetID, int relation){
            if(existID(sourceID) && existID(targetID)){
                Vertex* sourceV = &vertices.at(sourceID);
                Vertex* targetV = &vertices.at(targetID);
                sourceV->addEdge(targetV, relation);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist!";
            }
        }

        // function to check if the ID is present or not
        bool existID(unsigned long int id){
            if(vertices.find(id) == vertices.end()){
                return false;
            }
            return true;
        }

        // this function returns the total number of edges in the graph
        int getTotalNumberOfEdges(){
            int totalNumberEdges = 0;
            for(std::unordered_map<unsigned long int, Vertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                totalNumberEdges += it->second.getNumberOfEdges();
            }
            return totalNumberEdges;
        }

        // for debbuging
        void statistics(){
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << vertices.size() << std::endl;
            std::cout << "Number of relations/edges: " << getTotalNumberOfEdges() << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
        }

        // for debugging
        void printVertexByID(unsigned long int id){
            std::cout << "-------------- Vertex ID: " << id <<" --------------" << std::endl;
            Vertex* v = &vertices.at(id);
            std::cout << "Vertex-ID: \t"<< v->getId() << std::endl;
            std::cout << "LDBC-ID: \t"<< v->getLDBC_Id() << std::endl;
            std::cout << "Entity-ID: \t"<< v->getEntity() << std::endl;
            std::cout << "#Edges: \t" << v->getAdjList().size() << std::endl;
            std::cout << "Adj.List: ";

            const std::vector<Edge>& adjList = v->getAdjList();
            for(const auto& e : adjList){
                std::cout << "(" << e.target->getId() << "," << e.relation << ") ";
            }
            std::cout << "\n";
        }

    };

}

#endif //MORPHSTORE_GRAPH_H
