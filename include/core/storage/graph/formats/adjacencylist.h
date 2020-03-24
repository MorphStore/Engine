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

namespace morphstore{

    class AdjacencyList: public Graph {

    private:
        std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint64_t>>> adjacencylistPerVertex;

    public:
        storageFormat getStorageFormat() const override {
            return adjacencylist;
        }

        // function: to set graph allocations
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            vertices.reserve(numberVertices);
            adjacencylistPerVertex.reserve(numberVertices);
            edges.reserve(numberEdges);

            this->expectedEdgeCount = numberEdges;
            this->expectedVertexCount = numberVertices;
        }

        // function to add a single property to vertex
        void add_property_to_vertex(uint64_t id, const std::pair<std::string, std::string> property) override {
            if (exist_vertexId(id)) {
                vertices[id]->add_property(property);
            } else {
                std::cout << "Vertex with ID " << id << " not found." << std::endl;
            }
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

                for(const auto& edge : edgesToAdd) {
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
            if (adjacencylistPerVertex.find(id) == adjacencylistPerVertex.end()) {
                return 0;
            }
            else { 
                return adjacencylistPerVertex[id]->size();
            }
        }

        // get the neighbors-ids into vector for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
            std::vector<uint64_t> targetVertexIds = std::vector<uint64_t>();

            for(auto const edgeId: *adjacencylistPerVertex[id]) {
                targetVertexIds.push_back(edges[edgeId]->getTargetId());
            }
            
            return targetVertexIds;
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
            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<morphstore::Vertex>>);
            for(auto& it : vertices){
                // index size of vertex: size of id and sizeof pointer 
                index_size += sizeof(uint64_t) + sizeof(std::shared_ptr<morphstore::Vertex>);
                // data size:
                data_size += it.second->get_data_size_of_vertex();
            }

            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<morphstore::Edge>>);
            for(auto& it : edges){
                // index size of edge: size of id and sizeof pointer 
                index_size += sizeof(uint64_t) + sizeof(std::shared_ptr<morphstore::Edge>);
                // data size:
                data_size += it.second->size_in_bytes();
            }

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
                    auto edge = edges[edgeId];
                    std::cout << " Edge-ID: " << edge->getId() 
                              << " Type: " <<  get_edgeType_by_number(edge->getType())
                              << " Source-ID: " << edge->getSourceId() 
                              << " Target-ID: " << edge->getTargetId() 
                              << " Property: { "; 
                    edge->print_properties(); 
                    std::cout << std::endl << " }" << std::endl;
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
