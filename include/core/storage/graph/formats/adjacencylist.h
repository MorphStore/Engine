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

#include "../graph.h"
#include "../vertex/vertex.h"

#include <iterator>
#include <assert.h>

namespace morphstore{

    class AdjacencyList: public Graph {

    private:
        std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint64_t>>> adjacencylistPerVertex;

    public:
        AdjacencyList(VerticesContainerType vertices_container_type = VectorArrayContainer) : Graph(vertices_container_type) {}
        
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

            std::shared_ptr<std::vector<uint64_t>> adjacencyList;
            if (adjacencylistPerVertex.find(sourceId) != adjacencylistPerVertex.end()) {
                adjacencyList = adjacencylistPerVertex[sourceId];
            } else {
                adjacencyList = std::make_shared<std::vector<uint64_t>>();
                adjacencylistPerVertex[sourceId] = adjacencyList;
            }

            for(const auto edge : edgesToAdd) {
                edges[edge.getId()] = std::make_shared<Edge>(edge);
                if(vertices->exists_vertex(edge.getTargetId())) {
                    adjacencyList->push_back(edge.getId());
                }
                else {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
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
                    assert(edges.find(edgeId) != edges.end());
                    targetVertexIds.push_back(edges[edgeId]->getTargetId());
                }
            }
            
            return targetVertexIds;
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
