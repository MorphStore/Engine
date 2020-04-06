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
 * @file vertices__vectorvector_container.h
 * @brief storing vertices using a vector of vectors
 * @todo
*/

#ifndef MORPHSTORE_VERTICES_VECTORVECTOR_CONTAINER_H
#define MORPHSTORE_VERTICES_VECTORVECTOR_CONTAINER_H

#include "vertex.h"
#include "vertices_container.h"

#include <vector>
#include <utility>
#include <cstdlib>
#include <memory>

namespace morphstore{

    using vertex_vector_ptr = std::shared_ptr<std::vector<std::shared_ptr<morphstore::Vertex>>>;

    class VerticesVectorVectorContainer : public VerticesContainer{
        protected:
            std::vector<vertex_vector_ptr> vertices;
            uint64_t number_of_vertices = 0;
            vertex_vector_ptr current_vector;
            static const inline uint64_t vertex_vector_size = 4096;
            static const inline uint64_t vertices_per_vector = vertex_vector_size / Vertex::get_data_size_of_vertex();

             vertex_vector_ptr allocate_vertex_array() {
                 auto vertex_vector = std::make_shared<std::vector<std::shared_ptr<morphstore::Vertex>>>();
                 vertex_vector->reserve(vertex_vector_size / Vertex::get_data_size_of_vertex());
                 vertices.push_back(vertex_vector);

                 //std::cout << " Added a page" << std::endl;
                 //std::cout.flush();
                 
                 return vertex_vector;
            }

            inline uint64_t get_vertex_vector_number(uint64_t vertex_id) const {
                return vertex_id / vertex_vector_size;
            }

            inline uint64_t get_pos_in_vector(uint64_t vertex_id) const {
                return vertex_id % vertices_per_vector;
            }
            

            Vertex get_vertex_without_properties(uint64_t id) override {
                uint64_t vector_number = get_vertex_vector_number(id);
                uint64_t pos_in_vector = get_pos_in_vector(id);

                /*std::cout << " id: " << id 
                          << " vectors_number: " << vector_number 
                          << " pos in vector: " << pos_in_vector 
                          << " max_pos_in_vector: " << pos_in_vector 
                          << " max_pos_in_vector: " << vertices_per_vector 
                          << " number of vectors: " << vertices.size() << std::endl;          
                std::cout.flush();
                */

                assert (vector_number <= vertices.size());
                assert (pos_in_vector < vertices_per_vector);

                return *vertices.at(vector_number)->at(pos_in_vector);
            }
            

        public:
            void allocate(const uint64_t numberVertices) {
                VerticesContainer::allocate(numberVertices);
                current_vector = allocate_vertex_array();
            }

            void insert_vertex(Vertex v) {
                // equals current array is full
                if (current_vector->size() == vertices_per_vector) {
                    current_vector = allocate_vertex_array();
                }

                current_vector->push_back(std::make_shared<Vertex>(v));
                number_of_vertices++;;
            }

            bool exists_vertex(const uint64_t id) const override {
                // !assumes no deletion (should be replaced when an id-index exists)
                return number_of_vertices > id;
            }

            uint64_t vertex_count() const override {
                return number_of_vertices;
            }

            std::pair<size_t, size_t> get_size() override {
                auto [index_size, data_size] = VerticesContainer::get_size();
                
                // vector_count, current vertex_vector 
                index_size += 2 * sizeof(uint64_t);

                index_size += sizeof(std::vector<vertex_vector_ptr>);
                index_size += vertices.size() * sizeof(vertex_vector_ptr);
                
                for(auto vector: vertices) {
                    index_size += vector->size() * sizeof(std::shared_ptr<morphstore::Vertex>);
                    data_size += vector->size() * Vertex::get_data_size_of_vertex();
                }
                
                return std::make_pair(index_size, data_size);
            }
    };
}

#endif //MORPHSTORE_VERTICES_VECTORVECTOR_CONTAINER_H