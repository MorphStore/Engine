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
#include <core/storage/column.h>

#include <stdexcept>
#include <assert.h>

namespace morphstore{

    class CSR: public Graph{

    private:
        /* graph topology:
         * offset column: index is vertex-id; column entry contains offset in edgeId array
         * edgeId column: contains edge id
         */
        column<uncompr_f>* offset_column;
        column<uncompr_f>* edgeId_column;

    public:
        CSR(VerticesContainerType vertices_container_type = VectorArrayContainer) : Graph(vertices_container_type) {}

        std::string get_storage_format() const override {
            return "CSR";
        }

        // this function gets the number of vertices/edges and allocates memory for the vertices-map and the graph topology arrays
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            Graph::allocate_graph_structure(numberVertices, numberEdges);

            offset_column = column<uncompr_f>::create_global_column(numberVertices * sizeof(uint64_t));
            offset_column->set_count_values(numberVertices);
            edgeId_column = column<uncompr_f>::create_global_column(numberEdges * sizeof(uint64_t));
            edgeId_column->set_count_values(numberEdges);

            // init node array:
            uint64_t* offset_data = offset_column->get_data();
            offset_data[0] = 0;
        }

        // TODO: add a single edge in graph arrays -> needs a memory reallocating strategy
        void add_edge(uint64_t sourceId, uint64_t targetId, unsigned short int type) override {
            throw std::runtime_error("Singe edge addition not yet implemented for CSR" + sourceId + targetId + type);
        }

        // this function fills the graph-topology-arrays sequentially in the order of vertex-ids ASC
        // every vertex id contains a list of its neighbors
        void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge> edgesToAdd) override {
            assert(expectedEdgeCount >= getEdgeCount()+edgesToAdd.size());
            uint64_t* offset_data = offset_column->get_data();
            uint64_t offset = offset_data[sourceID];
            uint64_t nextOffset = offset + edgesToAdd.size();

            if (!vertices->exists_vertex(sourceID)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceID));
            }

            // fill the arrays
            // TODO: fill array using memcpy? (put edgeIds into vector as prerpare step)
            uint64_t* edgeId_data = edgeId_column->get_data();
            for(const auto& edge : edgesToAdd){
                std::shared_ptr<Edge> ePtr = std::make_shared<Edge>(edge);
                 if(!vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found " + edge.to_string());
                }
                edges[ePtr->getId()] = ePtr;
                
                edgeId_data[offset] = ePtr->getId();
                ++offset;
            }

            // to avoid buffer overflow:
            if(sourceID < getExpectedVertexCount()-1){
                offset_data[sourceID+1] = nextOffset;
            }
        }

        // get number of edges of vertex with id
        uint64_t get_out_degree(uint64_t id) override {
            uint64_t* offset_data = offset_column->get_data();
            uint64_t offset = offset_data[id];
            // special case: last vertex id has no next offset
            uint64_t nextOffset;

            // todo: `getExpectedVertexCount()` could be replaced by `offset_column->get_count_values()`
            if(id == getExpectedVertexCount() -1){
                nextOffset = getExpectedEdgeCount();
            }else{
                nextOffset = offset_data[id+1];
            }

            if(offset == nextOffset) return 0;
            uint64_t degree = nextOffset - offset;
            return degree;
        }

        // function to return a vector of ids of neighbors for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
             std::vector<uint64_t> neighbourEdgeIds;
             uint64_t* offset_data = offset_column->get_data();
             uint64_t offset = offset_data[id];
             uint64_t numberEdges = get_out_degree(id);

             // avoiding out of bounds ...
             // TODO: use assert here, as this is only out of bounds if the offset 
             if( offset < getExpectedEdgeCount()){
                 uint64_t* edgeId_data = edgeId_column->get_data();
                 neighbourEdgeIds.insert(neighbourEdgeIds.end(), edgeId_data+offset, edgeId_data+offset+numberEdges);
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

        void compress() override {
            std::cout << "Compressing graph format specific data structures";
            // TODO: need a way to change column format
        }

        // get size of storage format:
        std::pair<size_t, size_t> get_size_of_graph() const override {
            
            auto [index_size, data_size] = Graph::get_size_of_graph();

            index_size += edgeId_column->get_size_used_byte();
            index_size += offset_column->get_size_used_byte();

            return {index_size, data_size};
        }

        // for debugging:
        void print_neighbors_of_vertex(uint64_t id) override{
            std::cout << "Neighbours for Vertex with id " << id << std::endl;
            uint64_t* offset_data = offset_column->get_data();
            uint64_t offset = offset_data[id];
            uint64_t numberEdges = get_out_degree(id);
            
            uint64_t* edgeId_data = edgeId_column->get_data();
            for(uint64_t i = offset; i < offset+numberEdges; ++i){
                uint64_t edgeId = edgeId_data[i];
                print_edge_by_id(edgeId);
            }
        }
    };
}
#endif //MORPHSTORE_CSR_H
