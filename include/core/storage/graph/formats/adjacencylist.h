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
            if (exist_vertexId(sourceId)) {
                std::shared_ptr<std::vector<uint64_t>> adjacencyList;
                if (adjacencylistPerVertex.find(sourceId) != adjacencylistPerVertex.end()) {
                    adjacencyList = adjacencylistPerVertex[sourceId];
                } else {
                    adjacencyList = std::make_shared<std::vector<uint64_t>>();
                    adjacencylistPerVertex[sourceId] = adjacencyList;
                }

                for(const auto edge : edgesToAdd) {
                    edges[edge.getId()] = std::make_shared<Edge>(edge);
                    if(exist_vertexId(edge.getTargetId())) {
                        adjacencyList->push_back(edge.getId());
                    }
                    else {
                        std::cout << "Target-Vertex with ID " << edge.getTargetId() << " not found." << std::endl;
                    }
                }
            } else {
                std::cout << "Source-Vertex with ID " << sourceId << " not found." << std::endl;
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
        std::pair<size_t, size_t> get_size_of_graph() override {
            std::pair<size_t, size_t> index_data_size;
            
            auto [index_size, data_size] = Graph::get_size_of_graph();

            // adjacencyListPerVertex
            for(auto& it : adjacencylistPerVertex){
                // data size:
                data_size += sizeof(it);
            }

            index_data_size = {index_size, data_size};

            return index_data_size;
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
