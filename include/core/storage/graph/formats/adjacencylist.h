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
 * @brief Derived ADJ-List storage format class. Base: graph.h
 * @todo
*/

#ifndef MORPHSTORE_ADJACENCYLIST_H
#define MORPHSTORE_ADJACENCYLIST_H

#include "../graph.h"
#include "../vertex/adjacencylist_vertex.h"

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
        int add_vertex_with_properties(const std::unordered_map<std::string, std::string> &props) override {
            std::shared_ptr<Vertex> v = std::make_shared<AdjacencyListVertex>();
            v->setProperties(props);
            vertices[v->getID()] = v;
            return v->getID();
        }

        // function to add a single property to vertex
        void add_property_to_vertex(uint64_t id, const std::pair<std::string, const std::string> &property) override {
            if (exist_id(id)) {
                vertices[id]->add_property(property);
            } else {
                std::cout << "Vertex with ID " << id << " not found." << std::endl;
            }
        }

        // adding entity to vertex
        void add_entity_to_vertex(const uint64_t id, unsigned short int entity) override {
            if (exist_id(id)) {
                vertices[id]->setEntity(entity);
            } else {
                std::cout << "Vertex with ID " << id << " not found." << std::endl;
            }
        }

        // adding a single edge to vertex:
        void add_edge(uint64_t from, uint64_t to, unsigned short int rel) override {
            if (exist_id(from) && exist_id(to)) {
                vertices[from]->add_edge(from, to, rel);
            } else {
                std::cout << "Source-/Target-Vertex-ID does not exist in the database!" << std::endl;
            }
        }

        // function that adds multiple edges (list of neighbors) at once to vertex
        void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge>& relations) override {
            if (exist_id(sourceID)) {
                if (relations.size() != 0) {
                    vertices[sourceID]->add_edges(relations);
                }
            } else {
                std::cout << "Vertex with ID " << sourceID << " not found." << std::endl;
            }
        }

        void print_neighbors_of_vertex(uint64_t id) override{
            vertices[id]->print_neighbors();
        }

        // get number of neighbors of vertex with id
        uint64_t get_degree(uint64_t id) override {
            return vertices[id]->get_number_edges();
        }

        /* old-calculation of the graph size in bytes
        size_t get_size_of_graph(){
            size_t size = 0;
            size += sizeof(std::unordered_map<uint64_t, ADJLISTVertex>);
            for(std::unordered_map<uint64_t, ADJLISTVertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                size += it->second.get_size_of_vertex();
            }
            return size;
        }
         */

    };
}

#endif //MORPHSTORE_ADJACENCYLIST_H
