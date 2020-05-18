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

#include <core/storage/column_gen.h>
#include <core/storage/graph/graph.h>
#include <core/storage/graph/graph_compr_format.h>

#include <assert.h>
#include <iterator>
#include <type_traits>
#include <variant>

namespace morphstore {

    class AdjacencyList : public Graph {

    private:
        // const column as after finalized only read_only
        using adjacency_column = column_base *;
        using adjacency_vector = std::vector<uint64_t> *;
        using adjacency_list_variant = std::variant<adjacency_vector, adjacency_column>;

        struct Adjacency_List_Size_Visitor {
            size_t operator()(const adjacency_column c) const { return c->get_size_used_byte(); }
            size_t operator()(const adjacency_vector v) const { return sizeof(std::vector<uint64_t>) + (v->size() * sizeof(uint64_t)); }
        };

        struct Adjacency_List_OutDegree_Visitor {
            uint64_t operator()(const adjacency_column c) const {
                // assuming compressed col has the same value count (would not work for RLE)
                return c->get_count_values();
            }
            uint64_t operator()(const adjacency_vector v) const { 
                return v->size(); }
        };

        // maps the a list of outgoing edges (ids) to a vertex-id
        std::unordered_map<uint64_t, adjacency_list_variant> *adjacencylistPerVertex =
            new std::unordered_map<uint64_t, adjacency_list_variant>();

        // as default formats allocate to much memory for small columns
        // TODO: compress based on blocksize of format (as data smaller than blocksize gets not compressed?!)
        // TODO: as function parameter f.i. in change_min_compr_degree -> recall finalize and morph to current_compression
        static const uint64_t min_compr_degree = 100;

        // convert big-enough adj-vector to a (read-only) adj-column
        void finalize() {
            int vectors_transformed = 0;
            for (auto [id, adj_list] : *adjacencylistPerVertex) {
                if (std::holds_alternative<adjacency_vector>(adj_list)) {
                    auto adj_vector = std::get<adjacency_vector>(adj_list);
                    // this allows adding new edges to smaller adj_lists (even after morphing)
                    if (adj_vector->size() >= min_compr_degree) {
                        auto adj_col =
                            const_cast<column_uncompr *>(make_column(adj_vector->data(), adj_vector->size(), true));

                        (*adjacencylistPerVertex)[id] = adj_col;

                        // as v is not needed anymore and allocated using new
                        delete adj_vector;
                        vectors_transformed++;
                    }
                }
            }
#if DEBUG
            std::cout << "Transformed " << vectors_transformed << " vectors into columns" << std::endl;
#endif
        }

    public:
        ~AdjacencyList() {
            for (auto [id, adj_list] : *this->adjacencylistPerVertex) {
                if (std::holds_alternative<adjacency_column>(adj_list)) {
                    delete std::get<adjacency_column>(adj_list);
                } else {
                    delete std::get<adjacency_vector>(adj_list);
                }

                delete adjacencylistPerVertex;
            }
        }

        AdjacencyList(EdgesContainerType edges_container_type)
            : Graph(VerticesContainerType::VectorArrayContainer, edges_container_type) {}

        AdjacencyList(VerticesContainerType vertices_container_type = VerticesContainerType::VectorArrayContainer)
            : Graph(vertices_container_type) {}

        std::string get_storage_format() const override { return "Adjacency_List"; }

        // function: to set graph allocations
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            Graph::allocate_graph_structure(numberVertices, numberEdges);
            adjacencylistPerVertex->reserve(numberVertices);
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

            // avoid inserting an empty adjacencyVector (waste of memory)
            if (edgesToAdd.size() == 0) {
                return;
            }

            std::vector<uint64_t> *adjacencyVector;
            if (auto entry = adjacencylistPerVertex->find(sourceId); entry != adjacencylistPerVertex->end()) {
                if (std::holds_alternative<adjacency_column>(entry->second)) {
                    throw std::runtime_error("Not implemented to add edges, if adj. list is a (compressed) column");
                }

                adjacencyVector = std::get<adjacency_vector>(entry->second);
            } else {
                adjacencyVector = new std::vector<uint64_t>();
                adjacencylistPerVertex->insert({sourceId, adjacencyVector});
            }

            adjacencyVector->reserve(edgesToAdd.size());

            for (const auto edge : edgesToAdd) {
                if (!vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
                edges->add_edge(edge);
                adjacencyVector->push_back(edge.getId());
            }
        }

        // get number of neighbors of vertex with id
        uint64_t get_out_degree(uint64_t id) override {
            auto entry = adjacencylistPerVertex->find(id);
            if (entry == adjacencylistPerVertex->end()) {
                return 0;
            } else {
                return std::visit(Adjacency_List_OutDegree_Visitor{}, entry->second);
            }
        }

        std::vector<uint64_t> get_outgoing_edge_ids(uint64_t id) {
            // basically column -> vector (as convinient to use in other methods)
            // maybe better idea would be to return a uint64_t* instead (together with a size value)
            std::vector<uint64_t> edge_ids;
            if (auto entry = adjacencylistPerVertex->find(id); entry != adjacencylistPerVertex->end()) {
                auto adj_list = entry->second;
                if (std::holds_alternative<adjacency_column>(adj_list)) {
                    auto uncompr_col = decompress_graph_col(std::get<adjacency_column>(adj_list), current_compression);
                    const size_t column_size = uncompr_col->get_count_values();
                    // TODO: init vector via range-constructor / mem-cpy
                    // const uint8_t * end_addr = start_addr + sizeof(uint64_t) * out_degree;
                    const uint64_t *start_addr = uncompr_col->get_data();

                    edge_ids.insert(edge_ids.end(), start_addr, start_addr + column_size);

                    if (current_compression != GraphCompressionFormat::UNCOMPRESSED) {
                        delete uncompr_col;
                    }

                } else {
                    edge_ids = *std::get<adjacency_vector>(adj_list);
                }
            }

            return edge_ids;
        }

        // get the neighbors-ids into vector for BFS alg.
        // todo: this is actually format generic and can be pulled to graph.h
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
            std::vector<uint64_t> targetVertexIds;

            for (uint64_t const edgeId : get_outgoing_edge_ids(id)) {
                assert(edges->exists_edge(edgeId));
                targetVertexIds.push_back(edges->get_edge(edgeId).getTargetId());
            }

            return targetVertexIds;
        }

        // morphes the adj-lists to the given target_format
        // !!! first time overhead: as convert each vector to a column (finalizing) !!!
        void morph(GraphCompressionFormat target_format) override {
            // transform big enough vectors into columns
            this->finalize();

#if DEBUG
            std::cout << "Compressing graph format specific data structures using: " << graph_compr_f_to_string(target_format)
                      << std::endl;
            auto entry_count = adjacencylistPerVertex->size();
            int progress = 0;
#endif
            for (auto const [id, adj_list] : *adjacencylistPerVertex) {
#if DEBUG
                if (progress % 10000 == 0) {
                    std::cout << "Compression Progress: " << progress << "/" << entry_count << std::endl;
                }
                progress++;
#endif
                // currently min_compr_degree is final in adj_list and determines which adj-lists are
                // are columns (and not a vector)
                if (std::holds_alternative<adjacency_column>(adj_list)) {
                    auto old_adj_col = std::get<adjacency_column>(adj_list);
                    // const_cast needed as map-value is not constant
                    (*adjacencylistPerVertex)[id] = const_cast<adjacency_column>(
                        morph_graph_col(old_adj_col, current_compression, target_format, true));
                }
            }

            this->current_compression = target_format;
        }

        double compr_ratio() const {
            double total_compr_ratio = 0;
            for (auto const [id, adj_list] : *adjacencylistPerVertex) {
                auto out_degree = std::visit(Adjacency_List_OutDegree_Visitor{}, adj_list);
                double compr_ratio;
                if (std::holds_alternative<adjacency_column>(adj_list)) {
                    auto adj_col = std::get<adjacency_column>(adj_list);
                    compr_ratio = compression_ratio(adj_col, current_compression);
                } else {
                    compr_ratio = 1;
                }
                auto weighted_ratio = compr_ratio * ((double)out_degree / getEdgeCount());
                total_compr_ratio += weighted_ratio;
            }

            return total_compr_ratio;
        }

        double column_ratio() const {
            if (getEdgeCount() == 0) {
                return 1;
            }

            uint64_t column_count = 0;
            for (auto const [id, adj_list] : *adjacencylistPerVertex) {
                if (std::holds_alternative<adjacency_column>(adj_list)) {
                    column_count++;
                }
            }

            return (double)column_count / getEdgeCount();
        }

        // for measuring the size in bytes:
        std::pair<size_t, size_t> get_size_of_graph() const override {
            auto [index_size, data_size] = Graph::get_size_of_graph();

            // adjacencyListPerVertex
            index_size += sizeof(std::unordered_map<uint64_t, adjacency_list_variant>);
            index_size += adjacencylistPerVertex->size() * (sizeof(uint64_t) + sizeof(adjacency_list_variant));

            for (const auto [id, adj_list] : *adjacencylistPerVertex) {
                data_size += std::visit(Adjacency_List_Size_Visitor{}, adj_list);
            }

            return {index_size, data_size};
        }

        // for debugging: print neighbors a vertex
        void print_neighbors_of_vertex(uint64_t id) override {
            std::cout << std::endl << "Neighbours for Vertex with id " << id << std::endl;
            auto edge_ids = get_outgoing_edge_ids(id);

            if (edge_ids.size() == 0) {
                std::cout << "  No outgoing edges for vertex with id: " << id << std::endl;
            } else {
                for (const auto edge_id : edge_ids) {
                    print_edge_by_id(edge_id);
                }
            }
        }

        void statistics() override {
            Graph::statistics();
            std::cout << "Number of adjacency lists:" << adjacencylistPerVertex->size() << std::endl;
            std::cout << "Min. degree for compression: " << min_compr_degree << std::endl;
            std::cout << "Column ratio:" << column_ratio() << std::endl;
            std::cout << "Compression ratio:" << compr_ratio() << std::endl;
            std::cout << "--------------------------------------------" << std::endl;
            std::cout << std::endl << std::endl;
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_ADJACENCYLIST_H
