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
 * @todo Adjust get_size_of_graph(), ?replace unordered_map with a fixed sized array
*/

#ifndef MORPHSTORE_ADJACENCYLIST_H
#define MORPHSTORE_ADJACENCYLIST_H

#include <core/storage/graph/graph.h>

#include <iterator>
#include <assert.h>
#include <variant>
#include <type_traits>

namespace morphstore{

    class AdjacencyList: public Graph {

    private:
        using adjacency_column = column_base*; 
        using adjacency_vector = std::vector<uint64_t>*;
        using adjacency_list_variant = std::variant<adjacency_vector, adjacency_column>;

        struct Adjacency_List_Size_Visitor {
            size_t operator()(const adjacency_column c) const {
                return c->get_size_used_byte();
            }
            size_t operator()(const adjacency_vector v) const {
               return v->size();
            }
        };

        // maps the outgoing edges (ids) per vertex
        std::unordered_map<uint64_t, adjacency_list_variant> adjacencylistPerVertex;
        
        // indicating whether we have columns or vectors (columns after first compress() call)
        // TODO: is this replace-able by just checking the type of the first element in the map? (via holds_alternative)
        bool finalized = false;

        // convert every adjVector to a adjColumn
        void finalize() {
            if (!finalized) { 
                // use std::transform
            }
        }
    public:
        ~AdjacencyList() {
                for(auto entry: this->adjacencylistPerVertex) {
                    if (finalized) {
                        free(std::get<adjacency_column>(entry.second));
                    }
                    else {
                        free(std::get<adjacency_vector>(entry.second));
                    }
            }
        }

        AdjacencyList(EdgesContainerType edges_container_type)
            : Graph(VerticesContainerType::VectorArrayContainer, edges_container_type) {}

        AdjacencyList(VerticesContainerType vertices_container_type = VerticesContainerType::VectorArrayContainer) : Graph(vertices_container_type) {}
        
        std::string get_storage_format() const override {
            return "Adjacency_List";
        }

        // function: to set graph allocations
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            Graph::allocate_graph_structure(numberVertices, numberEdges);
            adjacencylistPerVertex.reserve(numberVertices);
        }

        // adding a single edge to vertex:
        void add_edge(uint64_t sourceId, uint64_t targetId, unsigned short int type) override {
            Edge e = Edge(sourceId, targetId, type);
            add_edges(sourceId, {e});
        }

        // function that adds multiple edges (list of neighbors) at once to vertex
        void add_edges(uint64_t sourceId, const std::vector<morphstore::Edge> edgesToAdd) override {
            if (!vertices->exists_vertex(sourceId)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceId));
            }
            // TODO: remove shared pointer?
            std::shared_ptr<std::vector<uint64_t>> adjacencyList;
            if (adjacencylistPerVertex.find(sourceId) != adjacencylistPerVertex.end()) {
                adjacencyList = adjacencylistPerVertex[sourceId];
            } else {
                adjacencyList = std::make_shared<std::vector<uint64_t>>();
                adjacencylistPerVertex[sourceId] = adjacencyList;
            }

            for (const auto edge : edgesToAdd) {
                if (!vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
                edges->add_edge(edge);
                adjacencyList->push_back(edge.getId());
            }
        }


        // get number of neighbors of vertex with id
        uint64_t get_out_degree(uint64_t id) override {
            auto entry = adjacencylistPerVertex.find(id);
            if (entry == adjacencylistPerVertex.end()) {
                return 0;
            }
            else { 
                return entry->second->size();
            }
        }

        // get the neighbors-ids into vector for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
            std::vector<uint64_t> targetVertexIds = std::vector<uint64_t>();

            auto entry = adjacencylistPerVertex.find(id);
            
            if (entry != adjacencylistPerVertex.end()) {
                for(uint64_t const edgeId: *(entry->second)) {
                    assert(edges->exists_edge(edgeId));
                    targetVertexIds.push_back(edges->get_edge(edgeId).getTargetId());
                }
            }
            
            return targetVertexIds;
        }

        void compress(GraphCompressionFormat target_format) override {
            std::cout << "Compressing graph format specific data structures using: " << to_string(target_format) << std::endl;

            //this->current_compression = target_format;
        }

        // for measuring the size in bytes:
        std::pair<size_t, size_t> get_size_of_graph() const override {
            auto [index_size, data_size] = Graph::get_size_of_graph();

            // adjacencyListPerVertex
            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint64_t>>>);
            index_size += adjacencylistPerVertex.size() * (sizeof(uint64_t) + sizeof(std::shared_ptr<std::vector<uint64_t>>));

            for(const auto& iterator : adjacencylistPerVertex){
                // might be wrong in case of compression
                data_size += sizeof(uint64_t) * iterator.second->size();
            }

            return {index_size, data_size};
        }

        // for debugging: print neighbors a vertex
        void print_neighbors_of_vertex(uint64_t id) override{
            std::cout << "Neighbours for Vertex with id " << id << std::endl;
            if(adjacencylistPerVertex.find(id) == adjacencylistPerVertex.end()) {
                std::cout << "  No outgoing edges for vertex with id: " << id << std::endl;
            }
            else {
                for (const auto edgeId : *adjacencylistPerVertex[id]) {
                    print_edge_by_id(edgeId);
                }
            }
        }

        void statistics() override {
            Graph::statistics();
            std::cout << "Number of adjacency lists:" << adjacencylistPerVertex.size() << std::endl;
            std::cout << std::endl << std::endl;
        }

    };
}

#endif //MORPHSTORE_ADJACENCYLIST_H
