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
 * @brief CSR vertex header file
 * @todo
*/

#ifndef MORPHSTORE_VERTEX_CSR_H
#define MORPHSTORE_VERTEX_CSR_H

#include <iostream>
#include <unordered_map>

namespace morphstore{

    class CSRVertex{

    private:
        // system-ID
        uint64_t id;
        // data 'properties'
        std::unordered_map<std::string, std::string> properties;
        // entity-number for look-up
        unsigned short int entity;

    public:

        CSRVertex(){
            // unique ID generation
            static uint64_t startID = 0;
            id = startID++;
        }

        // add entity to vertex
        void setEntity(unsigned short int e){
            this->entity = e;
        }

        unsigned short int getEntity(){
            return this->entity;
        }

        uint64_t getId() const{
            return id;
        }

        // calculate size of a vertex for memory usage in bytes
        size_t get_size_of_vertex() {
            size_t size = 0;
            size += sizeof(uint64_t); // id
            // properties:
            size += sizeof(std::unordered_map<std::string, std::string>);
            for(std::unordered_map<std::string, std::string>::iterator property = properties.begin(); property != properties.end(); ++property){
                size += sizeof(char)*(property->first.length() + property->second.length());
            }
            // entities:
            size += sizeof(unsigned short int);

            return size;
        }

        // this function adds a whole property map to a vertex
        void add_properties(const std::unordered_map<std::string, std::string> &properties){
            if(!properties.empty()){
                this->properties = properties;
            }else{
                std::cout << "The properties-list is empty!" << std::endl;
            }
        }

        // this adds one key-value pair to the vertex's property map
        void add_property(const std::pair<std::string, std::string>& property){
            this->properties[property.first] = std::move(property.second);
        }

        void print_properties(){
            for(const auto& entry : properties){
                std::cout << "{" << entry.first << ": " << entry.second << "}";
            }
        }
    };
}

#endif //MORPHSTORE_VERTEX_CSR_H
