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

#ifndef MORPHSTORE_GRAPH_ADJACENCY_LIST_H
#define MORPHSTORE_GRAPH_ADJACENCY_LIST_H

#include <core/storage/graph/adj_list/vertex.h>
#include <core/storage/graph/graph_abstract.h>

#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include <string>


namespace morphstore{

    class AdjacencyList: public morphstore::Graph{

    private:
        // main data structure: mapping global id -> vertex
        // unordered_map hast fast search time -> average = O(1); worst case = O(n):
        std::unordered_map<uint64_t, ADJLISTVertex> vertices;

        // lookup dictionaries for entities of vertices / relation names of edges
        std::map<unsigned short int, std::string> entityDictionary;
        std::map<unsigned short int, std::string> relationDictionary;

        const std::string storageFormat = "AdjacencyList";

    public:

        void init(){
            std::cout << "Nothing to do!!" << std::endl;
        }

        std::string getStorageFormat(){
            return storageFormat;
        }

        // calculate the graph size in bytes
        size_t get_size_of_graph(){
            size_t size = 0;
			size += sizeof(std::unordered_map<uint64_t, ADJLISTVertex>);
            for(std::unordered_map<uint64_t, ADJLISTVertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                size += it->second.get_size_of_vertex();
            }
            return size;
        }

        // adds a vertex (without properties)
        void add_vertex(){
            ADJLISTVertex v;
            vertices.insert(std::make_pair(v.getId(), v));
        }

        // function that creates a new relation/edge between two (existing) vertices
        void add_edge(const uint64_t sourceID, const uint64_t targetID, unsigned short int rel){
            if(exist_id(sourceID) && exist_id(targetID)){
                ADJLISTVertex* sourceV = &vertices.at(sourceID);
                ADJLISTVertex* targetV = &vertices.at(targetID);
                sourceV->add_edge(targetV, rel);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!";
            }
        }

        // function that creates a new relation/edge between two (existing) vertices WITH property
        void add_edge_with_property(uint64_t sourceID, uint64_t targetID, unsigned short int rel, const std::pair<std::string, std::string>& property){
            if(exist_id(sourceID) && exist_id(targetID)){
                ADJLISTVertex* sourceV = &vertices.at(sourceID);
                ADJLISTVertex* targetV = &vertices.at(targetID);
                sourceV->add_edge_with_property(targetV, rel, property);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!";
            }
        }

        // function to add a new (ldbc) vertex to the graph and returns system-ID
        uint64_t add_vertex_with_properties(const std::unordered_map<std::string, std::string>& props ){
            ADJLISTVertex v;
            v.add_properties(props);
            vertices.insert(std::make_pair(v.getId(), v));
            return v.getId();
        }

        // this adds a specific key-value pair (property) to a vertex given by its id
        void add_property_to_vertex(uint64_t id, const std::pair<std::string, const std::string>& property){
            if(exist_id(id)){
                vertices.at(id).add_property(property);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!" << std::endl;
            }
        }

        void add_entity_to_vertex(const uint64_t id, unsigned short int entity){
            if(exist_id(id)){
                vertices.at(id).setEntity(entity);
            }else{
                std::cout << "Vertex with ID " << id << " does not exist in the database!";
            }
        }

        std::string get_entity_by_number(unsigned short int e){
            if(entityDictionary.find( e ) != entityDictionary.end()){
                return entityDictionary.at(e);
            }else{
                return "No Matching of entity-number in the database!";
            }
        }

        void set_entity_dictionary(const std::map<unsigned short int, std::string>& entityList){
            this->entityDictionary = entityList;
        }

        std::string get_relation_by_number(unsigned short int re){
            if(relationDictionary.find( re ) != relationDictionary.end()){
                return relationDictionary.at(re);
            }else{
                return "No Matching of relation-number in the database!";
            }
        }

        void set_relation_dictionary(const std::map<unsigned short int, std::string>& relationList){
            this->relationDictionary = relationList;
        }

        // function to check if the ID is present or not
        bool exist_id(const uint64_t id){
            if(vertices.find(id) == vertices.end()){
                return false;
            }
            return true;
        }

        // this function returns the total number of edges in the graph
        uint64_t get_total_number_of_edges(){
            uint64_t totalNumberEdges = 0;
            for(std::unordered_map<uint64_t, ADJLISTVertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                totalNumberEdges += it->second.get_number_of_edges();
            }
            return totalNumberEdges;
        }

        // for debbuging
        void statistics(){
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << vertices.size() << std::endl;
            std::cout << "Number of relations/edges: " << get_total_number_of_edges() << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
        }

        // for debbuging
        void printEntities(){
            for(auto const& entity : entityDictionary){
                std::cout << entity.first << " -> " << entity.second << "\n";
            }
        }

        // for debbuging
        void printRelations(){
            for(auto const& rel : relationDictionary){
                std::cout << rel.first << " -> " << rel.second << "\n";
            }
        }

        // for debugging
        void print_vertex_by_id(uint64_t id){
            std::cout << "-------------- Vertex ID: " << id <<" --------------" << std::endl;
            ADJLISTVertex* v = &vertices.at(id);
            std::cout << "Vertex-ID: \t"<< v->getId() << std::endl;
            std::cout << "Entity: \t"<< get_entity_by_number(v->getEntity()) << std::endl;
            std::cout << "#Edges: \t" << v->get_adjList().size() << std::endl;
            std::cout << "Adj_List: ";

            const std::vector<Edge>& adjList = v->get_adjList();
            for(const auto& e : adjList){
                std::cout << "(" << e.target->getId() << "," << get_relation_by_number(e.relation) << ") ";
            }
            std::cout << "\n";
            std::cout << "Properties: "; v->print_properties();
            std::cout << "\n";
            std::cout << "-----------------------------------------------" << std::endl;
        }
    };

}

#endif //MORPHSTORE_GRAPH_ADJACENCY_LIST_H
