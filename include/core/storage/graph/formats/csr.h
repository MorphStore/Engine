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

#include <core/storage/graph/graph.h>

#include <stdexcept>
#include <assert.h>

namespace morphstore{

    class CSR: public Graph{

    private:
        /* graph topology:
         * offset column: index is vertex-id; column entry contains offset in edgeId array
         * edgeId column: contains edge id
         */
        column_base *offset_column;
        column_base *edgeId_column;

    public:
        ~CSR() {
            free(offset_column);
            free(edgeId_column);
        }

        CSR(EdgesContainerType edges_container_type)
            : Graph(VerticesContainerType::VectorArrayContainer, edges_container_type) {}

        CSR(VerticesContainerType vertices_container_type = VerticesContainerType::VectorArrayContainer)
            : Graph(vertices_container_type) {}

        std::string get_storage_format() const override {
            return "CSR";
        }

        // this function gets the number of vertices/edges and allocates memory for the vertices-map and the graph topology arrays
        // TODO: test that no data exists before (as this will get overwritten)
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            Graph::allocate_graph_structure(numberVertices, numberEdges);

            const size_t offset_size = numberVertices * sizeof(uint64_t);
            offset_column = new column<uncompr_f>(offset_size);
            offset_column->set_meta_data(numberVertices, offset_size);

            const size_t edge_ids_size = numberEdges * sizeof(uint64_t);
            edgeId_column = new column<uncompr_f>(edge_ids_size);
            edgeId_column->set_meta_data(numberEdges, edge_ids_size);

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
            // currently only read-only if compressed
            if (current_compression != GraphCompressionFormat::UNCOMPRESSED) {
                throw std::runtime_error("Edge insertion only allowed in uncompressed format. Current format: " + to_string(current_compression));
            }

            uint64_t* offset_data = offset_column->get_data();
            uint64_t offset = offset_data[sourceID];
            uint64_t nextOffset = offset + edgesToAdd.size();

            if (!vertices->exists_vertex(sourceID)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceID));
            }

            // fill the arrays
            // TODO: fill array using memcpy? (put edgeIds into vector as prepare step)
            uint64_t* edgeId_data = edgeId_column->get_data();
            for(const auto& edge : edgesToAdd){
                 if(!vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found " + edge.to_string());
                }
                edgeId_data[offset] = edge.getId();
                edges->add_edge(edge);
                ++offset;
            }

            // to avoid buffer overflow:
            if(sourceID < getExpectedVertexCount()-1){
                offset_data[sourceID+1] = nextOffset;
            }
        }

        // get number of edges of vertex with id
        uint64_t get_out_degree(uint64_t id) override {
            // decompressing offset_column in order to read correct offset
            // TODO: only decompress part of the column as only offset_column[id] and offset_column[id+1] will be read
            auto uncompr_offset_col = decompress_graph_col(offset_column, current_compression);
            uint64_t* offset_data = uncompr_offset_col->get_data();

            uint64_t offset = offset_data[id];
            uint64_t nextOffset;

            // special case: last vertex id has no next offset
            if(id == getVertexCount() -1){
                nextOffset = getEdgeCount();
            }else{
                nextOffset = offset_data[id+1];
            }

            // deleting temporary column
            if (uncompr_offset_col != offset_column) {
                delete uncompr_offset_col;
            }

            // compute out_degree
            if (offset == nextOffset)
                return 0;
            else {
                return nextOffset - offset;
            }
        }

        // function to return a vector of ids of neighbors for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
            std::vector<uint64_t> targetVertexIds;
            for (auto edge_id : get_outgoing_edge_ids(id)) {
                assert(edges->exists_edge(edge_id));
                targetVertexIds.push_back(edges->get_edge(edge_id).getTargetId());
            }

            return targetVertexIds;
        }

        void compress(GraphCompressionFormat target_format) override {
            std::cout << "Morphing graph format specific data structures from "
                      << to_string(current_compression) << " to " << to_string(target_format) << std::endl;

            if (current_compression == target_format) {
                std::cout << "Already in " << to_string(target_format);
                return;
            }

            offset_column = const_cast<column_base*>(morph_graph_col(offset_column, current_compression, target_format, true));
            edgeId_column = const_cast<column_base*>(morph_graph_col(edgeId_column, current_compression, target_format, true));

            std::cout << " offset col compression ratio: "
                      << compression_ratio(offset_column, target_format) << std::endl
                      << " edgeId col compression ratio: "
                      << compression_ratio(edgeId_column, target_format) << std::endl;

            this->current_compression = target_format;
        }

        // get size of storage format:
        std::pair<size_t, size_t> get_size_of_graph() const override {
            
            auto [index_size, data_size] = Graph::get_size_of_graph();

            index_size += edgeId_column->get_size_used_byte();
            index_size += offset_column->get_size_used_byte();

            return {index_size, data_size};
        }

        std::vector<uint64_t> get_outgoing_edge_ids(uint64_t id) {
            assert(vertices->exists_vertex(id));

            std::vector<uint64_t> out_edge_ids;
            auto uncompr_offset_col = decompress_graph_col(offset_column, current_compression);
            uint64_t offset = ((uint64_t *)uncompr_offset_col->get_data())[id];

            if (uncompr_offset_col != offset_column) {
                delete uncompr_offset_col;
            }

            // TODO: decompressing offset_column twice this way
            uint64_t numberEdges = get_out_degree(id);

            auto uncompr_edgeId_col = decompress_graph_col(edgeId_column, current_compression);
            uint64_t *edgeId_data = uncompr_edgeId_col->get_data();
            out_edge_ids.insert(out_edge_ids.end(), edgeId_data + offset, edgeId_data + offset + numberEdges);

            if (uncompr_edgeId_col != edgeId_column) {
                delete uncompr_edgeId_col;
            }

            return out_edge_ids;
        }

        // for debugging:
        // TODO: simply by using a get_outgoing_edges(id) method
        void print_neighbors_of_vertex(uint64_t id) override{
            std::cout << "Neighbours for Vertex with id " << id << std::endl;

            for(auto const edge_id: get_outgoing_edge_ids(id)){
                print_edge_by_id(edge_id);
            }
        }

        std::string get_column_info(const column_base *column) {
            return " values: " + std::to_string(column->get_count_values()) + " size in bytes: " + std::to_string(column->get_size_used_byte());
        }

        void statistics() override {
            Graph::statistics();
            std::cout << "offset column: " << get_column_info(offset_column) << std::endl;
            std::cout << "edgeId column: " << get_column_info(edgeId_column) << std::endl;
        }
    };
}
#endif //MORPHSTORE_CSR_H
