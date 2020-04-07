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
 * @file vertices__hashmap_container.h
 * @brief storing vertices using a hashmap
 * @todo
*/

#ifndef MORPHSTORE_VERTICES_HASHMAP_CONTAINER_H
#define MORPHSTORE_VERTICES_HASHMAP_CONTAINER_H

#include "vertex.h"
#include "vertices_container.h"

#include <map>
#include <unordered_map>

namespace morphstore{

    class VerticesHashMapContainer : public VerticesContainer{
        protected:
            std::unordered_map<uint64_t , std::shared_ptr<morphstore::Vertex>> vertices;

            Vertex get_vertex_without_properties(uint64_t id) override{
                return *vertices[id];
            }

        public:
            void allocate(const uint64_t numberVertices) override {
                VerticesContainer::allocate(numberVertices);
                this->vertices.reserve(numberVertices);
            }
            
            void insert_vertex(const Vertex v) override {
                vertices[v.getID()] = std::make_unique<Vertex>(v);
            }

            bool exists_vertex(const uint64_t id) const override {
                if(vertices.find(id) == vertices.end()){
                    return false;
                }
                return true;
            }

            uint64_t vertex_count() const {
                return vertices.size();
            }

            std::pair<size_t, size_t> get_size() const override {
                auto [index_size, data_size] = VerticesContainer::get_size();

                // container for indexes:
                index_size += sizeof(std::unordered_map<uint64_t, std::unique_ptr<Vertex>>);
                // index size of vertex: size of id and sizeof pointer
                index_size += vertices.size() * (sizeof(uint64_t) + sizeof(std::unique_ptr<Vertex>));
                data_size += vertices.size() * Vertex::get_data_size_of_vertex();
                

                return {index_size, data_size};
            }
    };
}

#endif //MORPHSTORE_VERTICES_HASHMAP_CONTAINER_H