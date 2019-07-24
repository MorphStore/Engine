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

#include <unordered_map>
#include <map>
#include <vector>
#include <string>

namespace morphstore{

    class CSR{

    private:
        // main data structure: hash table (hash id to vertex)
        // unordered_map has fast search time / look-up -> average = O(1); worst case = O(n):
        std::unordered_map<uint64_t, CSRVertex> vertices;

        // graph-structure: 3 Arrays (row_array, col_array, val_array)
        // row array('node array'): contains the offset in the col_array; vertex-system-id is index in the row_array
        // col_array('edge array'): every cell represents an edge containing the vertex targets ID
        // value_array: relation number
        uint64_t* node_array = nullptr;
        uint64_t* edge_array = nullptr;
        unsigned short int* val_array = nullptr;

        // lookup dictionaries for entities of vertices / relation names of edges
        std::map<unsigned short int, std::string> entityDictionary;
        std::map<unsigned short int, std::string> relationDictionary;

        uint64_t numberEdges;

    public:

        ~CSR(){
            delete [] node_array;
            delete [] edge_array;
            delete [] val_array;
        }

        // this functions allocates the memory for the graph structure arrays
        void allocate_graph_structure_memory(uint64_t numberVertices, uint64_t numberEdges){

            // allocate node array:
            node_array = new uint64_t[numberVertices];

            // allocate edge array:
            edge_array = new uint64_t[numberEdges];
            setNumberEdges(numberEdges);

            // allocate val array:
            val_array = new unsigned short int[numberEdges];
        }

        std::string getStorageFormat() const{
            return "CSR";
        }

        uint64_t getNumberEdges(){
            return numberEdges;
        }

        void setNumberEdges(uint64_t edges){
            this->numberEdges = edges;
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

        // this function adds the data in the CSR structure from LDBC-Importer
        void add_edge_ldbc(uint64_t vertexID, uint64_t startOffset, const std::vector<std::pair<uint64_t , unsigned short int >>& neighbors){
            node_array[vertexID] = startOffset; // offset in edge_array
            for(auto const& pair : neighbors){
                edge_array[startOffset] = pair.first; // target id
                val_array[startOffset]  = pair.second; // relation number for lookup
                ++startOffset;
            }
        }

        void add_edge_with_property(uint64_t sourceID, uint64_t targetID, unsigned short int relation, const std::pair<std::string, std::string>& property){
            // TODO IMPLEMENT
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

        uint64_t getNumberVertices(){
            return vertices.size();
        }

        // calculate the graph size in bytes
        size_t get_size_of_graph(){
            size_t size = 0;
            // pointer to arrays:
            size += sizeof(uint64_t*) * 2 + sizeof(unsigned short int*);
            // vertices:
            size += sizeof(uint64_t) * getNumberVertices();
            // edges:
            size += sizeof(uint64_t) * getNumberEdges();
            // val array:
            size += sizeof(unsigned short int) * getNumberEdges();

            // vertex map wth actual data:
            for(std::unordered_map<uint64_t, CSRVertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                size += it->second.get_size_of_vertex();
            }

            return size;
        }


        // for debugging
        void print_vertex_by_id(uint64_t id){
            std::cout << "-------------- Vertex ID: " << id <<" --------------" << std::endl;
            CSRVertex* v = &vertices.at(id);
            uint64_t startOffset = node_array[id];
            uint64_t endOffset = node_array[id+1];
            std::cout << "Offset: " << startOffset << std::endl;
            std::cout << "Vertex-ID: \t"<< v->getId() << std::endl;
            std::cout << "Entity: \t"<< get_entity_by_number(v->getEntity()) << std::endl;
            std::cout << "#Edges: " << (endOffset-startOffset) << std::endl;
            std::cout << "Relations: ";
            for (uint64_t i = startOffset; i < endOffset; ++i) {
                std::cout << "(" << edge_array[i] << "," << val_array[i] << "." << get_relation_by_number(val_array[i]) << ")  ";
            }
            std::cout << "\n";
            std::cout << "Properties: "; v->print_properties();
            std::cout << "\n";
            std::cout << "-----------------------------------------------" << std::endl;
        }


        // for debbuging
        void statistics(){
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << getNumberVertices() << std::endl;
            std::cout << "Number of relations/edges: " << getNumberEdges() << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
        }
    };

}

#endif //MORPHSTORE_GRAPH_CSR_H
