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
 * @file vertex.h
 * @brief abstract vertex class for storage formats
 * @todo
*/

#ifndef MORPHSTORE_VERTEX_H
#define MORPHSTORE_VERTEX_H

#include "../edge/edge.h"

#include <unordered_map>
#include <iostream>
#include <vector>
#include <memory>

namespace morphstore{

    class Vertex{

    protected:
        // vertex: id,
        uint64_t id;      
        // optional: type, properties
        unsigned short int type;


    public:
        Vertex(uint64_t id, unsigned short int type){
            this->id = id;
            this->type = type;
        }

        uint64_t getID(){
            return id;
        }

        unsigned short getType() const {
            return type;
        }

         // get size of vertex in bytes:
        size_t get_data_size_of_vertex() {
            size_t size = 0;
            size += sizeof(uint64_t); // id
            size += sizeof(unsigned short int); // entity

            return size;
        }
    };

    // convinience class for returning whole vertices
    class VertexWithProperties {
        private:
            std::shared_ptr<Vertex> vertex;
            std::unordered_map<std::string, std::string> properties;
        public:
            VertexWithProperties(std::shared_ptr<Vertex> vertex, const std::unordered_map<std::string, std::string> properties) {
                this->vertex = vertex;
                this->properties = properties;
            }

            uint64_t getID() {
                return vertex->getID();
            }

            unsigned short getType() const {
                return vertex->getType();
            }

            std::unordered_map<std::string, std::string> getProperties() {
                return properties;
            }
    };

}

#endif //MORPHSTORE_VERTEX_H
