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
 * @file edges__hashmap_container.h
 * @brief storing edges using a hashmap
 * @todo an EntityHashMapContainer abstraction (reduce duplicated code to VertexHashMapContainer)
 */

#ifndef MORPHSTORE_EDGES_HASHMAP_CONTAINER_H
#define MORPHSTORE_EDGES_HASHMAP_CONTAINER_H

#include "edge.h"
#include "edges_container.h"

#include <map>
#include <unordered_map>

namespace morphstore {

    class EdgesHashMapContainer : public EdgesContainer {
    protected:
        // mapping edge id -> edge
        // currently saving the id twice 
        std::unordered_map<uint64_t, EdgeWithId> edges;

    public:
        std::string container_description() const override { return "unordered_map<uint64_t , EdgeWithId>"; }

        void allocate(const uint64_t expected_edges) override {
            EdgesContainer::allocate(expected_edges);
            this->edges.reserve(expected_edges);
        }

        // TODO: unpack EdgeWithId to just Edge (avoid saving edge-id twice)
        void insert_edge(const EdgeWithId e) override { edges[e.getId()] = e; }

        bool exists_edge(const uint64_t id) const override {
            if (edges.find(id) == edges.end()) {
                return false;
            }
            return true;
        }

        EdgeWithId get_edge(uint64_t id) override { return edges[id]; }

        uint64_t edge_count() const { return edges.size(); }

        // memory estimation 
        // returns a pair of index-size, data-size
        std::pair<size_t, size_t> get_size() const override {
            auto [index_size, data_size] = EdgesContainer::get_size();

            // container for indexes:
            index_size += sizeof(std::unordered_map<uint64_t, EdgeWithId>);
            // index size of edge: size of id and sizeof pointer
            index_size += edges.size() * sizeof(uint64_t);
            data_size += edges.size() * EdgeWithId::size_in_bytes();

            return {index_size, data_size};
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_EDGES_HASHMAP_CONTAINER_H