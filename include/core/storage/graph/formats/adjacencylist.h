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
 * @file adjacencylist.h
 * @brief Derived adj. list storage format class. Base: graph.h
 * @todo
*/

#ifndef MORPHSTORE_ADJACENCYLIST_H
#define MORPHSTORE_ADJACENCYLIST_H

#include "../graph.h"
#include "../vertex/adjacencylist_vertex.h"

#include <iterator>

namespace morphstore{

    class AdjacencyList: public Graph {

    public:

        storageFormat getStorageFormat() const override {
            return adjacencylist;
        }

        // function: to set graph allocations
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            vertices.reserve(numberVertices);
            setNumberEdges(numberEdges);
            setNumberVertices(numberVertices);
        }

        // adding a single vertex
        void add_vertex() override {
            std::shared_ptr<Vertex> v = std::make_shared<AdjacencyListVertex>();
            vertices[v->getID()] = v;
        }

        // adding a vertex with its properties
        uint64_t add_vertex_with_properties(const std::unordered_map<std::string, std::string> props) override {
            std::shared_ptr<Vertex> v = std::make_shared<AdjacencyListVertex>();
            v->setProperties(props);
            vertices[v->getID()] = v;
            return v->getID();
        }

        // function to add a single property to vertex
        void add_property_to_vertex(uint64_t id, const std::pair<std::string, std::string> property) override {
            if (exist_id(id)) {
                vertices[id]->add_property(property);
            } else {
                std::cout << "Vertex with ID " << id << " not found." << std::endl;
            }
        }

        // adding type to vertex
        void add_type_to_vertex(const uint64_t id, const unsigned short int type) override {
            if (exist_id(id)) {
                vertices[id]->setType(type);
            } else {
                std::cout << "Vertex with ID " << id << " not found." << std::endl;
            }
        }

        // adding a single edge to vertex:
        void add_edge(uint64_t from, uint64_t to, unsigned short int type) override {
            if (exist_id(from) && exist_id(to)) {
                vertices[from]->add_edge(from, to, type);
            } else {
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!" << std::endl;
            }
        }

        // function that adds multiple edges (list of neighbors) at once to vertex
        void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge> relations) override {
            if (exist_id(sourceID)) {
                if (relations.size() != 0) {
                    vertices[sourceID]->add_edges(relations);
                }
            } else {
                std::cout << "Vertex with ID " << sourceID << " not found." << std::endl;
            }
        }

        // for debugging: print neighbors a vertex
        void print_neighbors_of_vertex(uint64_t id) override{
            vertices[id]->print_neighbors();
        }

        // get number of neighbors of vertex with id
        uint64_t get_degree(uint64_t id) override {
            return vertices[id]->get_number_edges();
        }

        // get the neighbors-ids into vector for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
            return vertices.at(id)->get_neighbors_ids();
        }

        // for measuring the size in bytes:
        std::pair<size_t, size_t> get_size_of_graph() override {
            std::pair<size_t, size_t> index_data_size;
            size_t data_size = 0;
            size_t index_size = 0;

            // lookup type dicts
            index_size += 2 * sizeof(std::map<unsigned short int, std::string>);
            for(auto& ent : vertexTypeDictionary){
                index_size += sizeof(unsigned short int);
                index_size += sizeof(char)*(ent.second.length());
            }
            for(auto& rel : edgeTypeDictionary){
                index_size += sizeof(unsigned short int);
                index_size += sizeof(char)*(rel.second.length());
            }

            // container for indexes:
            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<morphstore::AdjacencyListVertex>>);
            for(auto& it : vertices){
                // index size of vertex: size of id and sizeof pointer 
                index_size += sizeof(uint64_t) + sizeof(std::shared_ptr<morphstore::AdjacencyListVertex>);
                // data size:
                data_size += it.second->get_data_size_of_vertex();
            }

            index_data_size = {index_size, data_size};

            return index_data_size;
        }

    };
}

#endif //MORPHSTORE_ADJACENCYLIST_H
