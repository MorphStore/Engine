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
 * @brief base graph class for any storage format --> CSR,ADJ
 * @todo
*/

#ifndef MORPHSTORE_GRAPH_H
#define MORPHSTORE_GRAPH_H

#include "vertex/vertex.h"
#include "vertex/vertices_hashmap_container.h"
#include "vertex/vertices_vectorvector_container.h"
#include "vertex/vertices_vectorarray_container.h"
#include "edge/edge.h"
#include "property_type.h"

#include <map>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <fstream>
#include <sstream>
#include <assert.h>


namespace morphstore{

    class Graph{

    protected:
        uint64_t expectedVertexCount;
        uint64_t expectedEdgeCount;

        mutable uint64_t currentMaxVertexId = 0;

        // ! currently need to change to right container (abstract seems not to be possible due to pure virtual functions)
        VerticesVectorArrayContainer vertices;

        std::unordered_map<uint64_t , std::shared_ptr<Edge>> edges;

        std::unordered_map<uint64_t, std::unordered_map<std::string, property_type>> edge_properties;


        // Lookup for types: number to string
        std::map<unsigned short int, std::string> edgeTypeDictionary;


        // function to check if the edge-ID is present or not (exists)
        bool exist_edgeId(const uint64_t id){
            if(edges.find(id) == edges.end()){
                return false;
            }
            return true;
        }
        
        // TODO: put this into vertex container?
        uint64_t getNextVertexId() const {
            return currentMaxVertexId++;
        }

    public:
        // -------------------- Setters & Getters --------------------

        void set_vertex_type_dictionary(const std::map<unsigned short, std::string>& types) {
            assert(types.size() != 0);
            this->vertices.set_vertex_type_dictionary(types);
        }

        const std::map<unsigned short, std::string> &getRelationDictionary() const {
            return edgeTypeDictionary;
        }

        void setEdgeTypeDictionary(const std::map<unsigned short, std::string>& rel) {
            this->edgeTypeDictionary = rel;
        }

        uint64_t getExpectedVertexCount() const {
            return expectedVertexCount;
        }

        uint64_t getVertexCount() const {
            return vertices.vertex_count();
        }

        uint64_t getExpectedEdgeCount() const {
            return expectedEdgeCount;
        }

        uint64_t getEdgeCount() const {
            return edges.size();
        }

        uint64_t add_vertex(const unsigned short int type, const std::unordered_map<std::string, property_type> props = {}) {
            assert(expectedVertexCount > getVertexCount());
            Vertex v = Vertex(getNextVertexId(), type);
            vertices.add_vertex(v, props);  
            return v.getID();
        };

        std::string get_edgeType_by_number(unsigned short int type){
            if(edgeTypeDictionary.find( type ) != edgeTypeDictionary.end()){
                return edgeTypeDictionary.at(type);
            }else{
                print_type_dicts();
                return std::to_string(type) + " not found in edge-type dictionary";
            }
        }

        // function which returns a pointer to vertex by id
        VertexWithProperties get_vertex(uint64_t id){
            return vertices.get_vertex(id);
        }

        // function which returns a pointer to vertex by id
        EdgeWithProperties get_edge(uint64_t id){
            return EdgeWithProperties(edges[id], edge_properties[id]);
        }

	    // function to return a list of pair < vertex id, degree > DESC:
        std::vector<std::pair<uint64_t, uint64_t>> get_list_of_degree_DESC(){
            std::vector<std::pair<uint64_t, uint64_t>> vertexDegreeList;
            vertexDegreeList.reserve(expectedVertexCount);
            // fill the vector with every vertex key and his degree
            for(uint64_t i = 0; i < expectedVertexCount; ++i){
                vertexDegreeList.push_back({i, this->get_out_degree(i)});
            }
            // sort the vector on degree DESC
            std::sort(vertexDegreeList.begin(), vertexDegreeList.end(), [](const std::pair<uint64_t, uint64_t> &left, const std::pair<uint64_t ,uint64_t> &right) {
                return left.second > right.second;
            });

            return vertexDegreeList;
        }

        // function to measure graph characteristics (degree and count):
        void measure_degree_count(std::string filePath){
            std::vector<std::pair<uint64_t, uint64_t>> verticesDegree = get_list_of_degree_DESC();
            // unordered map for mapping degree to count:
	        std::unordered_map<uint64_t, uint64_t> results;

	        for(uint64_t i = 0; i < verticesDegree.size(); ++i){
		        // increment count in results for a given degree:
	    	    results[verticesDegree[i].second]++;
	        }

	        // write to file:
	        std::ofstream fs;
            std::stringstream ss;
            // open file for writing and delete existing stuff:
            fs.open(filePath, std::fstream::out | std::ofstream::trunc);

            for(auto const& m : results){
                ss << m.first << "," << m.second << "\n";
            }
            fs << ss.str() ;
            fs.close();
        }

        void add_property_to_vertex(uint64_t id, const std::pair<std::string, property_type> property) {
            vertices.add_property_to_vertex(id, property);
        };

        void add_properties_to_edge(uint64_t id, const std::unordered_map<std::string, property_type> properties) {
            edge_properties[id] = properties;
        };

        // -------------------- pure virtual functions --------------------

        virtual std::string get_storage_format() const = 0;
        virtual void add_edge(uint64_t from, uint64_t to, unsigned short int rel) = 0;
        virtual void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge> relations) = 0;
        virtual uint64_t get_out_degree(uint64_t id) = 0;
        virtual std::vector<uint64_t> get_neighbors_ids(uint64_t id) = 0;

	    virtual std::pair<size_t, size_t> get_size_of_graph(){
            // including vertices + its properties + its type dict
            auto [index_size, data_size] = vertices.get_size();

            // lookup type dicts
            for(auto& rel : edgeTypeDictionary){
                index_size += sizeof(unsigned short int);
                index_size += sizeof(char)*(rel.second.length());
            }

            // container for indexes:
            index_size += sizeof(std::vector<std::vector<std::shared_ptr<morphstore::Vertex>>>);

            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<morphstore::Edge>>);
            for(auto& it : edges){
                // index size of edge: size of id and sizeof pointer 
                index_size += sizeof(uint64_t) + sizeof(std::shared_ptr<morphstore::Edge>);
                // data size:
                data_size += it.second->size_in_bytes();
            }

            // TODO: extra propertymappings class 
            // edge-properties:
            index_size += sizeof(std::unordered_map<uint64_t, std::unordered_map<std::string, std::string>>);
            for(auto& property_mapping: edge_properties) {
                index_size += sizeof(uint64_t) + sizeof(std::unordered_map<std::string, std::string>);
                for (std::unordered_map<std::string, property_type>::iterator property = property_mapping.second.begin(); property != property_mapping.second.end(); ++property) {
                    data_size += sizeof(char) * (property->first.length() + sizeof(property->second));
                }
            }

            return std::make_pair(index_size, data_size);
        };

        virtual void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) {
            this->expectedVertexCount = numberVertices;
            this->expectedEdgeCount = numberEdges;

            vertices.allocate(numberVertices);

            edges.reserve(numberEdges);
            edge_properties.reserve(numberEdges);
        };

        // -------------------- debugging functions --------------------

        // for debugging
        virtual void print_neighbors_of_vertex(uint64_t id) = 0;

        virtual void statistics(){
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << getVertexCount() << std::endl;
            std::cout << "Number of vertices with properties:" << vertices.vertices_with_properties_count() << std::endl;
            std::cout << "Number of edges: " << getEdgeCount() << std::endl;
            std::cout << "Number of edges with properties:" << edge_properties.size() << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
        }

        void print_vertex_by_id(uint64_t id) {
            vertices.print_vertex_by_id(id);
            std::cout << "\n";
            std::cout << "#Edges: " << this->get_out_degree(id);
            std::cout << "\n";
            std::cout << "-----------------------------------------------" << std::endl;
        }

        void print_edge_by_id(uint64_t id) {
            std::cout << "-------------- Edge ID: " << id << " --------------" << std::endl;
            std::shared_ptr<Edge> edge = edges[id];
            std::cout << "Edge-ID: \t" << edge->getId() << std::endl;
            std::cout << "Source-ID: \t" << edge->getSourceId() << std::endl;
            std::cout << "Target-ID: \t" << edge->getTargetId() << std::endl;
            std::cout << "Type: \t" << get_edgeType_by_number(edge->getType()) << std::endl;
            std::cout << "\n";
            std::cout << "Properties: ";
            for (const auto entry  : edge_properties[id]) {
                auto value = entry.second;
                std::cout << "{" << entry.first << ": "; 
                std::visit(PropertyValueVisitor{}, value);
                std::cout << "}";
            }
            std::cout << "\n";
            std::cout << "-----------------------------------------------" << std::endl;
        }

        void print_type_dicts(){
            vertices.print_type_dict();

            std::cout << "EdgeType-Dict: " << std::endl;
            for(auto const& rel : edgeTypeDictionary){
                std::cout << rel.first << " -> " << rel.second << std::endl;
            }
        }

    };

}


#endif //MORPHSTORE_GRAPH_H
