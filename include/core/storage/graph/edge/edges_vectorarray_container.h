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
 * @file edges__vectorarray_container.h
 * @brief storing edges using a vector of arrays; assuming a consecutive id space
 * @todo
 */

#ifndef MORPHSTORE_EDGES_VECTORARRAY_CONTAINER_H
#define MORPHSTORE_EDGES_VECTORARRAY_CONTAINER_H

#include "edge.h"
#include "edges_container.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace morphstore {
    // very different to VerticesVectorArrayContainer as edge ids are not given at insertion time! 
    // (not anymore, but not considered in current implementation)
    // and using std::array as aligned_alloc did not set invalid flag to false (could be solveable)
    class EdgesVectorArrayContainer : public EdgesContainer {
    protected:
        static const inline uint64_t edge_array_size = 4096;
        static const inline uint64_t edges_per_array = edge_array_size / sizeof(EdgeWithId);

        using edge_array = std::array<EdgeWithId, edges_per_array>;
        std::vector<edge_array> edges;

        uint64_t number_of_edges = 0;

        edge_array allocate_edge_array() {
            edge_array array;
            edges.push_back(array);
            // std::cout << " Added a page" << std::endl;
            // std::cout.flush();

            return array;
        }

        inline uint64_t get_edge_array_number(uint64_t edge_id) const { return edge_id / edges_per_array; }

        inline uint64_t get_pos_in_array(uint64_t edge_id) const { return edge_id % edges_per_array; }

    public:
        std::string container_description() const override {
            return "vector<array<EdgeWithId, " + std::to_string(edges_per_array) + ">>";
        }

        void allocate(const uint64_t expected_edges) override {
            EdgesContainer::allocate(expected_edges);
            // rounding up ..  only whole arrays can be allocated 
            auto array_count = std::ceil(expected_edges / (float)edges_per_array);
            this->edges.reserve(array_count);

            for (int i = 0; i < array_count; i++) {
                allocate_edge_array();
            }
        }

        void insert_edge(EdgeWithId e) {
            // not assuming sequentiell insertion (could be changed to just insert at a given position)
            // and only assert that the given position matches
            auto array_number = get_edge_array_number(e.getId());
            auto array_pos = get_pos_in_array(e.getId());
            
            // second time to assert that expected edge count is not exceeded ?
            if (array_number >= edges.size()) {
                throw std::runtime_error("Exceeded edge id limit: Edge id " + std::to_string(e.getId()) + " > " +
                                         std::to_string(edges_per_array * edges.size() - 1));
            }

            /* if (edges.at(array_number)[array_pos].isValid()) {
                throw std::runtime_error("Delete existing edge before overwriting it: edge-id " + e.to_string());
            } */

            edges.at(array_number)[array_pos] = e;
            number_of_edges++;
        }

        bool exists_edge(const uint64_t id) const override {
            uint64_t array_number = get_edge_array_number(id);
            uint64_t pos_in_array = get_pos_in_array(id);

            if (array_number >= edges.size())
                return false;

            return edges.at(array_number)[pos_in_array].isValid();
        }

        EdgeWithId get_edge(uint64_t id) override {
            uint64_t array_number = get_edge_array_number(id);
            uint64_t pos_in_array = get_pos_in_array(id);

            assert(array_number < edges.size());

            return edges.at(array_number)[pos_in_array];
        }

        uint64_t edge_count() const override { return number_of_edges; }

        // memory estimation 
        // returns a pair of index-size, data-size
        std::pair<size_t, size_t> get_size() const override {
            auto [index_size, data_size] = EdgesContainer::get_size();

            // vector count, current_array_offset
            index_size += 2 * sizeof(uint64_t);

            index_size += sizeof(std::vector<edge_array>);
            // allocated memory for edges
            data_size += edges.size() * sizeof(edge_array);

            return {index_size, data_size};
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_EDGES_VECTORARRAY_CONTAINER_H