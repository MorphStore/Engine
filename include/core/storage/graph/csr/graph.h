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
 * @brief CSR graph header file
 * @todo
*/

#ifndef MORPHSTORE_GRAPH_CSR_H
#define MORPHSTORE_GRAPH_CSR_H

#include <core/storage/graph/csr/vertex.h>
#include <core/storage/graph/graph_abstract.h>

#include <unordered_map>
#include <map>
#include <string>

namespace morphstore{

    class CSR: public morphstore::Graph{

    private:
        // main data structure: hash table (hash id to vertex)
        // unordered_map has fast search time / look-up -> average = O(1); worst case = O(n):
        std::unordered_map<uint64_t, CSRVertex> vertices;

        // graph-structure: 3 Arrays (row_array, col_array, val_array)
        // row array('node array'): contains the offset in the col_array; vertex-system-id is index in the row_array
        // col_array('edge array'): every cell represents an edge containing the vertex targets ID
        // value_array: edge properties
        uint64_t* node_array;
        uint64_t* edge_array;
        std::string* val_array;

        // lookup dictionaries for entities of vertices / relation names of edges
        std::map<unsigned short int, std::string> entityDictionary;
        std::map<unsigned short int, std::string> relationDictionary;

        const std::string storageFormat = "CSR";


    public:

        std::string getStorageFormat(){
            return storageFormat;
        }

        // this functions allocates the memory for the graph structure arrays
        void init(){
            // (1) get number of vertices from main data structure
            uint64_t numberVertices = vertices.size();
            // (2) allocate node array memory
            node_array = new uint64_t[numberVertices];
        }

        void add_vertex(){
            CSRVertex v;
            vertices.insert(std::make_pair(v.getId(), v));
        }

        void add_property_to_vertex(uint64_t id, const std::pair<std::string, const std::string>& property){
            if(exist_id(id)){
                vertices.at(id).add_property(property);
            }else{
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!" << std::endl;
            }
        }

        void add_edge(const uint64_t sourceID, const uint64_t targetID, unsigned short int relation){
            // TODO
            std::cout << sourceID << targetID << relation << std::endl;
        }

        void add_edge_with_property(uint64_t sourceID, uint64_t targetID, unsigned short int relation, const std::pair<std::string, std::string>& property){
            // TODO
            std::cout << sourceID << targetID << relation << property.first << std::endl;

        }

        uint64_t add_vertex_with_properties(const std::unordered_map<std::string, std::string>& props ){
            CSRVertex v;
            v.add_properties(props);
            vertices.insert(std::make_pair(v.getId(), v));
            return v.getId();
        }

        void add_entity_to_vertex(const uint64_t id, unsigned short int entity){
            if(exist_id(id)){
                vertices.at(id).setEntity(entity);
            }else{
                std::cout << "Vertex with ID " << id << " does not exist in the database!";
            }
        }

        // function to check if the ID is present or not
        bool exist_id(const uint64_t id){
            if(vertices.find(id) == vertices.end()){
                return false;
            }
            return true;
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



        // for debbuging
        void statistics(){
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << vertices.size() << std::endl;
            std::cout << "Number of relations/edges: " << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
        }


    };

}

#endif //MORPHSTORE_GRAPH_CSR_H
