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
 * @file csr.h
 * @brief Derived CSR storage format class. Base: graph.h
 * @todo Edge_value_array should only store edge-ids (not whole objects)
*/

#ifndef MORPHSTORE_CSR_H
#define MORPHSTORE_CSR_H

#include "../graph.h"
#include "../vertex/vertex.h"
#include <stdexcept>
#include <assert.h>

namespace morphstore{

    class CSR: public Graph{

    private:
        /* graph topology:
         * offset array: index is vertex-id; array cell contains offset in edgeId array
         * edgeId array: contains edge id
         */
        uint64_t* offset_array = nullptr;
        uint64_t* edgeId_array = nullptr;

    public:

        ~CSR() {
            free(offset_array);
            free(edgeId_array);
        }

        std::string get_storage_format() const override {
            return "CSR";
        }

        // this function gets the number of vertices/edges and allocates memory for the vertices-map and the graph topology arrays
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            Graph::allocate_graph_structure(numberVertices, numberEdges);

            offset_array = (uint64_t*) malloc(numberVertices * sizeof(uint64_t));
            edgeId_array = (uint64_t*) malloc(numberEdges * sizeof(uint64_t));

            // init node array:
            offset_array[0] = 0;
        }

        // TODO: add a single edge in graph arrays -> needs a memory reallocating strategy
        void add_edge(uint64_t sourceId, uint64_t targetId, unsigned short int type) override {
            throw std::runtime_error("Singe edge addition not yet implemented for CSR" + sourceId + targetId + type);
        }

        // this function fills the graph-topology-arrays sequentially in the order of vertex-ids ASC
        // every vertex id contains a list of its neighbors
        void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge> edgesToAdd) override {
            assert(expectedEdgeCount >= getEdgeCount()+edgesToAdd.size());
            uint64_t offset = offset_array[sourceID];
            uint64_t nextOffset = offset + edgesToAdd.size();

            if (!vertices.exists_vertex(sourceID)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceID));
            }

            // fill the arrays
            for(const auto& edge : edgesToAdd){
                std::shared_ptr<Edge> ePtr = std::make_shared<Edge>(edge);
                 if(!vertices.exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found " + edge.to_string());
                }
                edges[ePtr->getId()] = ePtr;
                edgeId_array[offset] = ePtr->getId();
                ++offset;
            }

            // to avoid buffer overflow:
            if(sourceID < getExpectedVertexCount()-1){
                offset_array[sourceID+1] = nextOffset;
            }
        }

        // get number of edges of vertex with id
        uint64_t get_out_degree(uint64_t id) override {
            uint64_t offset = offset_array[id];
            // special case: last vertex id has no next offset
            uint64_t nextOffset;
            if(id == getExpectedVertexCount() -1){
                nextOffset = getExpectedEdgeCount();
            }else{
                nextOffset = offset_array[id+1];
            }

            if(offset == nextOffset) return 0;
            uint64_t degree = nextOffset - offset;
            return degree;
        }

        // function to return a vector of ids of neighbors for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
             std::vector<uint64_t> neighbourEdgeIds;
             uint64_t offset = offset_array[id];
             uint64_t numberEdges = get_out_degree(id);

             // avoiding out of bounds ...
             if( offset < getExpectedEdgeCount()){
                 neighbourEdgeIds.insert(neighbourEdgeIds.end(), edgeId_array+offset, edgeId_array+offset+numberEdges);
             }

             std::vector<uint64_t> targetVertexIds;

             // resolving each edgeId
             for (auto edgeId: neighbourEdgeIds)
             {
                 assert(edges.find(edgeId) != edges.end());
                 targetVertexIds.push_back(edges[edgeId]->getTargetId());
             }
             
             return targetVertexIds;
        }

        // get size of storage format:
        std::pair<size_t, size_t> get_size_of_graph() override {
            std::pair<size_t, size_t> index_data_size;
            
            auto [index_size, data_size] = Graph::get_size_of_graph();

            // pointer to arrays:
            index_size += sizeof(uint64_t*) * 2 + sizeof(Edge*);
            // edges array values:
            for(uint64_t i = 0; i < getExpectedEdgeCount(); i++){
                index_size += sizeof(uint64_t); // node_array with offsets
            }

            index_data_size = {index_size, data_size};

            return index_data_size;
        }

        // for debugging:
        void print_neighbors_of_vertex(uint64_t id) override{
            std::cout << "Neighbours for Vertex with id " << id << std::endl;
            uint64_t offset = offset_array[id];
            uint64_t numberEdges = get_out_degree(id);

            for(uint64_t i = offset; i < offset+numberEdges; ++i){
                uint64_t edgeId = edgeId_array[i];
                print_edge_by_id(edgeId);
            }
        }
    };
}
#endif //MORPHSTORE_CSR_H
