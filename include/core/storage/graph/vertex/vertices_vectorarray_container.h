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
 * @file vertices__vectorarray_container.h
 * @brief storing vertices using a vector of arrays
 * @todo
*/

#ifndef MORPHSTORE_VERTICES_VECTORARRAY_CONTAINER_H
#define MORPHSTORE_VERTICES_VECTORARRAY_CONTAINER_H

#include "vertex.h"
#include "vertices_container.h"

#include <vector>
#include <cstdlib>

namespace morphstore{

    class VerticesVectorArrayContainer : public VerticesContainer{
        protected:
            std::vector<Vertex*> vertices;

            static const inline uint64_t vertex_array_size = 4096;
            static const inline uint64_t vertices_per_array = vertex_array_size / sizeof(Vertex);

            uint64_t number_of_vertices = 0;
            Vertex* current_array;
            uint64_t current_array_offset = 0;


            Vertex* allocate_vertex_array() {
                auto array_pointer = (Vertex *) std::aligned_alloc(
                    sizeof(Vertex), 
                    vertices_per_array * sizeof(Vertex));

                vertices.push_back(array_pointer);
                //std::cout << " Added a page" << std::endl;
                //std::cout.flush();
                return array_pointer;
            }

            inline uint64_t get_vertex_vector_number(uint64_t vertex_id) const {
                return vertex_id / vertex_array_size;
            }

            inline uint64_t get_pos_in_array(uint64_t vertex_id) const {
                return vertex_id % vertices_per_array;
            }

            Vertex get_vertex_without_properties(uint64_t id) override {
                uint64_t array_number = get_vertex_vector_number(id);
                uint64_t pos_in_array = get_pos_in_array(id);

                //assert (pos_in_array < vertices_per_array);
                //assert (array_number < vertices.size());

                return vertices.at(array_number)[pos_in_array];
            }         

        public:
            // TODO: make array_size based on constructor
            //VerticesVectorArrayContainer(array_size) 

            ~VerticesVectorArrayContainer() {
                // TODO: find memory leak (destructor seems not to be called)
                std::cout << "freeing vertex pages";
                for (auto array_pointer : this->vertices) {
                    free(array_pointer);
                }
            }

            std::string container_description() const override {
                return "vector<Vertex*>";
            }

            void allocate(const uint64_t numberVertices) override {
                VerticesContainer::allocate(numberVertices);
                this->vertices.reserve(number_of_vertices / vertices_per_array);
                current_array = allocate_vertex_array();
            }

            void insert_vertex(Vertex v) {
                // equals current array is full
                if (current_array_offset == vertices_per_array) {
                    current_array = allocate_vertex_array();
                    current_array_offset = 0;
                }

                current_array[current_array_offset] = v;
                current_array_offset++;
                number_of_vertices++;
            }

            bool exists_vertex(const uint64_t id) const override {
                // assumes no deletion! else retrieve vertrex at position and check isValid()
                return number_of_vertices > id;
            }

            uint64_t vertex_count() const override {
                return number_of_vertices;
            }

            std::pair<size_t, size_t> get_size() const override {
                auto [index_size, data_size] = VerticesContainer::get_size();

                // vector count, current_array_offset 
                index_size += 2 * sizeof(uint64_t);
                // current_array
                index_size += sizeof(Vertex*);
                index_size += sizeof(std::vector<Vertex*>);
                index_size += vertices.size() * sizeof(Vertex*);
                // allocated memory for vertices
                data_size  += vertices.size() * Vertex::get_data_size_of_vertex() * vertices_per_array;
                
                return {index_size, data_size};
            }
    };
}

#endif //MORPHSTORE_VERTICES_VECTORARRAY_CONTAINER_H