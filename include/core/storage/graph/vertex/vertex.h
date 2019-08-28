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
 * @todo add vertex size calculation
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
        // optional: entity, properties
        uint64_t id;
        unsigned short int entity;
        std::unordered_map<std::string, std::string> properties;


    public:

        // ----------------- Setter & Getter -----------------

        uint64_t getID(){
            return id;
        }

        unsigned short getEntity() const {
            return entity;
        }

        void setEntity(unsigned short e) {
            Vertex::entity = e;
        }

        const std::unordered_map<std::string, std::string> &getProperties() const {
            return properties;
        }

        void setProperties(const std::unordered_map<std::string, std::string> &props) {
            Vertex::properties = props;
        }

        // function that adds a single property key-value pair to vertex
        void add_property(const std::pair<std::string, std::string>& property){
            /*
            auto it = properties.find(property.first);
            if(it != properties.end()){
                it->second = property.second;
            }
             */
            this->properties[property.first] = std::move(property.second);
        }


        // ----------------- (pure) virtual functions -----------------
        virtual void add_edges(const std::vector<morphstore::Edge>& edges) = 0;
        virtual void add_edge(uint64_t from, uint64_t to, unsigned short int rel) = 0;
        virtual void print_neighbors() = 0;
        virtual size_t get_size_of_vertex() = 0;

        virtual uint64_t get_number_edges(){
            return 0;
        };

        // for BFS alg.: adj-list
        virtual std::vector<uint64_t> get_neighbors_ids() {
            // return empty vector: implementation only needed in ADj-Vertex
            return std::vector<uint64_t>();
        }


        // ----------------- DEBUGGING -----------------
        void print_properties() {
            for (const auto &entry : properties) {
                std::cout << "{" << entry.first << ": " << entry.second << "}";
            }
            std::cout << "\n";
        }
    };

}

#endif //MORPHSTORE_VERTEX_H
