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

namespace morphstore{

    class Vertex{

    protected:
        // vertex: id,
        uint64_t id;      
        // optional: type, properties
        unsigned short int type;
        std::unordered_map<std::string, std::string> properties;


    public:

        // ----------------- Setter & Getter -----------------

        Vertex(uint64_t id){
            this->id = id;
        }

        uint64_t getID(){
            return id;
        }

        unsigned short getType() const {
            return type;
        }

        void setType(const unsigned short type) {
            Vertex::type = type;
        }

        const std::unordered_map<std::string, std::string> &getProperties() const {
            return properties;
        }

        void setProperties(const std::unordered_map<std::string, std::string> props) {
            Vertex::properties = props;
        }

        // function that adds a single property key-value pair to vertex
        void add_property(const std::pair<std::string, std::string> property){
            this->properties[property.first] = property.second;//std::move(property.second);
        }

         // get size of vertex in bytes:
        size_t get_data_size_of_vertex() {
            size_t size = 0;
            size += sizeof(uint64_t); // id
            size += sizeof(unsigned short int); // entity
            // properties:
            size += sizeof(std::unordered_map<std::string, std::string>);
            for(std::unordered_map<std::string, std::string>::iterator property = properties.begin(); property != properties.end(); ++property){
                size += sizeof(char)*(property->first.length() + property->second.length());
            }

            return size;
        }

        // ----------------- DEBUGGING -----------------
        void print_properties() {
            for (const auto entry  : properties) {
                std::cout << "{" << entry.first << ": " << entry.second << "}";
            }
            std::cout << "\n";
        }
    };

}

#endif //MORPHSTORE_VERTEX_H
