/**********************************************************************************************
 * Copyright (C) 2020 by MorphStore-Team                                                      *
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
#include <core/morphing/graph/morph_saving_offsets_graph_col.h>

#include <assert.h>
#include <stdexcept>

namespace morphstore {

    class CSR : public Graph {

    private:
        /* graph topology:
         * offset column: index is vertex-id; column entry contains offset in edgeId array
         * edgeId column: contains edge id
         */
        column_with_blockoffsets_base *offset_column;
        column_with_blockoffsets_base *edgeId_column;

    protected:
        // this function fills the graph-topology-arrays sequentially in the order of vertex-ids ASC
        void add_to_vertex_edges_mapping(uint64_t sourceID, const std::vector<uint64_t> edge_ids) override {
            // TODO: throw error if not in order of vertex-ids ASC inserted (currently will only produce rubbish data)
            // TODO: handle if sourceIDs are skipped
            // potential solution: add last_seen_vertex_id as class field .. check based on that .. assert order and
            // insert offsets for skipped vertices

            // avoid writting more than reserved (as fixed sized columns)
            assert(expectedEdgeCount >= getEdgeCount());

            // currently only read-only if compressed
            if (current_compression != GraphCompressionFormat::UNCOMPRESSED) {
                throw std::runtime_error("Edge insertion only allowed in uncompressed format. Current format: " +
                                         graph_compr_f_to_string(current_compression));
            }

            uint64_t *offset_data = offset_column->get_column()->get_data();
            uint64_t offset = offset_data[sourceID];
            uint64_t nextOffset = offset + edge_ids.size();

            uint64_t *edgeId_data = edgeId_column->get_column()->get_data();
            // TODO: get copy to work (should be faster than loop)
            // std::copy(edge_ids.begin(), edge_ids.end(), edgeId_data);
            for (auto edge_id : edge_ids) {
                edgeId_data[offset] = edge_id;
                offset++;
            }

            // to avoid buffer overflow:
            if (sourceID < getExpectedVertexCount() - 1) {
                offset_data[sourceID + 1] = nextOffset;
            }
        }

        uint64_t get_offset(uint64_t id) const {
            // TODO: use cache

            auto block_size = offset_column->get_block_size();

            if (current_compression == GraphCompressionFormat::UNCOMPRESSED) {
                uint64_t *col_data = offset_column->get_column()->get_data();
                return col_data[id];
            } else {
                auto block_number = id / block_size;
                auto block_pos = id % block_size;

                assert(block_number < offset_column->get_block_offsets()->size());

                auto uncompr_block = decompress_column_block(offset_column, current_compression, block_number);

                uint64_t *block_data = uncompr_block->get_data();

                auto offset = block_data[block_pos];

                // deleting temporary column
                delete uncompr_block;

                return offset;
            }
        }

        // DEBUG function to look into column:
        void print_column(const column_base *col, int start, int end) const {
            // validate interval (fix otherwise)
            int col_size =  col->get_count_values();
            if (start < 0 || col_size < start) {
                start = 0;
            }
            if (col_size <= end) {
                end = col->get_count_values() - 1;
            }

            std::cout << "Printing column from " << start << " to " << end << std::endl;
            const uint64_t *data = col->get_data();

            for (auto pos = start; pos <= end; pos++) {
                std::cout << "Index: " << pos << " Value:" << data[pos] << std::endl;
            }
        }

    public:
        ~CSR() {
            delete offset_column;
            delete edgeId_column;
        }

        CSR(EdgesContainerType edges_container_type)
            : Graph(VerticesContainerType::VectorArrayContainer, edges_container_type) {}

        CSR(VerticesContainerType vertices_container_type = VerticesContainerType::VectorArrayContainer)
            : Graph(vertices_container_type) {}

        std::string get_storage_format() const override { return "CSR"; }

        // this function gets the number of vertices/edges and allocates memory for the vertices-map and the graph
        // topology arrays
        // TODO: test that no data exists before (as this will get overwritten)
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            Graph::allocate_graph_structure(numberVertices, numberEdges);

            const size_t offset_size = numberVertices * sizeof(uint64_t);
            auto offset_col = new column<uncompr_f>(offset_size);
            offset_col->set_meta_data(numberVertices, offset_size);
            // wrapping offset_column
            offset_column = new column_with_blockoffsets<uncompr_f>(offset_col);

            const size_t edge_ids_size = numberEdges * sizeof(uint64_t);
            auto edgeId_col = new column<uncompr_f>(edge_ids_size);
            edgeId_col->set_meta_data(numberEdges, edge_ids_size);
            // wrapping edgeId_column
            edgeId_column = new column_with_blockoffsets<uncompr_f>(edgeId_col);

            // init node array:
            uint64_t *offset_data = offset_col->get_data();
            offset_data[0] = 0;
        }

        // TODO: add a single edge in graph arrays -> needs a memory reallocating strategy
        uint64_t add_edge(uint64_t sourceId, uint64_t targetId, unsigned short int type) override {
            throw std::runtime_error("Singe edge addition not yet implemented for CSR" + sourceId + targetId + type);
        }

        // get number of edges of vertex with id
        uint64_t get_out_degree(uint64_t id) const override {
            uint64_t offset = get_offset(id);
            uint64_t nextOffset;

            // special case: last vertex id has no next offset
            if (id == getVertexCount() - 1) {
                nextOffset = getEdgeCount();
            } else {
                nextOffset = get_offset(id + 1);
            }

            // compute out_degree
            // TODO: simplify this line
            if (offset == nextOffset)
                return 0;
            else {
                return nextOffset - offset;
            }
        }

        std::vector<uint64_t> get_outgoing_edge_ids(uint64_t id) const override {
            assert(vertices->exists_vertex(id));

            std::vector<uint64_t> result;

            uint64_t start = get_offset(id);
            uint64_t degree = get_out_degree(id);

            // TODO: use cache
            result.reserve(degree);

            // end is not included in the result
            auto end = start + degree;

            assert(start <= end);
            assert(getEdgeCount() >= end);

            if (degree == 0) {
                return result;
            }

            auto block_size = edgeId_column->get_block_size();

            if (current_compression == GraphCompressionFormat::UNCOMPRESSED) {
                uint64_t *col_data = edgeId_column->get_column()->get_data();
                result.insert(result.end(), col_data + start, col_data + end);
            } else {
                // getting one block at a time as most of the time only one block is needed
                // also allows to use block cache (would need to inject cache into decompress_block otherwise)
                auto start_block = start / block_size;
                auto start_block_pos = start % block_size;
                auto end_block = end / block_size;
                auto end_block_pos = end % block_size;

                 assert(start_block < edgeId_column->get_block_offsets()->size());
                 assert(end_block < edgeId_column->get_block_offsets()->size());

                // case that end is the first value of another block (should not decompress that block than)
                if (end_block_pos == 0) {
                    end_block--;
                    end_block_pos = block_size + 1;
                }

                for (auto block_number = start_block; block_number <= end_block; block_number++) {
                    auto uncompr_block = decompress_column_block(edgeId_column, current_compression, block_number);
                    uint64_t *block_data = uncompr_block->get_data();

                    // all edge ids in the same block
                    if (start_block == end_block) {
                        result.insert(result.end(), block_data + start_block_pos, block_data + end_block_pos);
                    } else if (block_number == end_block) {
                        // only insert until end_pos
                        result.insert(result.end(), block_data, block_data + end_block_pos);
                    } else if (block_number == start_block) {
                        // don't insert values before start
                        auto block_end = block_data + block_size;
                        result.insert(result.end(), block_data + start_block_pos, block_end);
                    } else {
                        // insert whole block (should be very rare)
                        auto block_end = block_data + block_size;
                        result.insert(result.end(), block_data, block_end);
                    }

                    // deleting temporary column
                    delete uncompr_block;
                }
            }

            assert(result.size() == degree);

            return result;
        }

        void morph(GraphCompressionFormat target_format) override {
#if DEBUG
            std::cout << "Morphing graph format specific data structures from "
                      << graph_compr_f_to_string(current_compression) << " to "
                      << graph_compr_f_to_string(target_format) << std::endl;
#endif
            if (current_compression == target_format) {
#if DEBUG
                std::cout << "Already in " << graph_compr_f_to_string(target_format);
#endif
                return;
            }

            offset_column = morph_saving_offsets_graph_col(offset_column, current_compression, target_format, true);
            edgeId_column = morph_saving_offsets_graph_col(edgeId_column, current_compression, target_format, true);

            this->current_compression = target_format;
        }

        // get size of storage format:
        std::pair<size_t, size_t> get_size_of_graph() const override {

            auto [index_size, data_size] = Graph::get_size_of_graph();
            
            // column_meta_data, prepared_for_random_access, .. not included in get_size_used_byte;
            index_size += 2 * sizeof(column<uncompr_f>);
            index_size += edgeId_column->get_size_used_byte();
            index_size += offset_column->get_size_used_byte();

            return {index_size, data_size};
        }

        double offset_column_compr_ratio() { return compression_ratio(offset_column, current_compression); }

        double edgeId_column_compr_ratio() { return compression_ratio(edgeId_column, current_compression); }

        std::string get_column_info(column_with_blockoffsets_base *col_with_offsets) {
            auto col = col_with_offsets->get_column();

            return " values: " + std::to_string(col->get_count_values()) +
                   " size in bytes: " + std::to_string(col->get_size_used_byte()) +
                   " compression ratio: " + std::to_string(compression_ratio(col_with_offsets, current_compression)) +
                   " number of blocks (if blocksize > 1): " + std::to_string(col_with_offsets->get_block_offsets()->size());
        }

        void statistics() override {
            Graph::statistics();
            std::cout << "offset column: " << get_column_info(offset_column) << std::endl;
            std::cout << "edgeId column: " << get_column_info(edgeId_column) << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
            std::cout << std::endl << std::endl;
        }
    };
} // namespace morphstore
#endif // MORPHSTORE_CSR_H
