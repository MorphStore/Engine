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
 * @file vertices_container.h
 * @brief abstract class for storing vertices
 * @todo
*/

#ifndef MORPHSTORE_VERTICES_CONTAINER_H
#define MORPHSTORE_VERTICES_CONTAINER_H

#include "vertex.h"
#include "../property_type.h"

#include <map>
#include <unordered_map>
#include <assert.h>
#include <utility>

namespace morphstore{
    enum class VerticesContainerType {HashMapContainer, VectorArrayContainer};
    class VerticesContainer {
        protected:
            uint64_t currentMaxVertexId = 0; 
            uint64_t expected_vertex_count = 0;
            std::map<unsigned short int, std::string> vertex_type_dictionary;

            // TODO: try other property storage formats than per node .. (triple-store or per property)
            std::unordered_map<uint64_t, std::unordered_map<std::string, property_type>> vertex_properties;

            std::string get_vertex_type(unsigned short int type) const {
                if (vertex_type_dictionary.find(type) != vertex_type_dictionary.end()) {
                    return vertex_type_dictionary.at(type);
                }
                else {
                    return "No Matching of type-number in the database! For type " + std::to_string(type);
                }
            }

            uint64_t getNextVertexId() {
                return currentMaxVertexId++;
            }

        public:
            virtual std::string container_description() const = 0;
            virtual void insert_vertex(Vertex v) = 0;
            virtual bool exists_vertex(const uint64_t id) const = 0;
            virtual Vertex get_vertex(uint64_t id) = 0;
            virtual uint64_t vertex_count() const = 0;


            virtual void allocate(uint64_t expected_vertices) {
                vertex_properties.reserve(expected_vertices);
                expected_vertex_count += expected_vertices;
            }

            uint64_t add_vertex(const unsigned short int type, const std::unordered_map<std::string, property_type> properties = {}) {
                assert(currentMaxVertexId < expected_vertex_count);
                Vertex v = Vertex(getNextVertexId(), type);
                insert_vertex(v);
                if (!properties.empty()) {
                    vertex_properties.insert(std::make_pair(v.getID(), properties));
                }

                return v.getID();
            }

            void add_property_to_vertex(uint64_t id, const std::pair<std::string, property_type> property) {
                assert(exists_vertex(id));
                vertex_properties[id].insert(property);
            };

            void set_vertex_type_dictionary(const std::map<unsigned short, std::string>& types) {
                assert(types.size() != 0);
                this->vertex_type_dictionary = types;
            }
            

            const VertexWithProperties get_vertex_with_properties(uint64_t id) {
                assert(exists_vertex(id));
                return VertexWithProperties(get_vertex(id), vertex_properties[id]);
            }

            uint64_t vertices_with_properties_count() {
                return vertex_properties.size();
            }

            virtual std::pair<size_t, size_t> get_size() const {
                size_t data_size = 0;
                size_t index_size = 0;

                // lookup type dicts
                index_size += 2 * sizeof(std::map<unsigned short int, std::string>);
                for(auto& type_mapping : vertex_type_dictionary){
                    index_size += sizeof(unsigned short int);
                    index_size += sizeof(char)*(type_mapping.second.length());
                }

                // vertex-properties: 
                index_size += sizeof(std::unordered_map<uint64_t, std::unordered_map<std::string, std::string>>);
                for (const auto &property_mapping : vertex_properties) {
                    index_size += sizeof(uint64_t) + sizeof(std::unordered_map<std::string, std::string>);
                    for (const auto &property : property_mapping.second) {
                        data_size += sizeof(char) * property.first.length() + sizeof(property.second);
                    }
                }

                return {index_size, data_size};
            }

            void print_type_dict(){
                std::cout << "VertexType-Dict: " << std::endl;
                for (auto const &entry : vertex_type_dictionary) {
                    std::cout << entry.first << " -> " << entry.second << std::endl;
                }
            }

            void print_vertex_by_id(const uint64_t id) {
                std::cout << "-------------- Vertex ID: " << id << " --------------" << std::endl;
                VertexWithProperties v = get_vertex_with_properties(id);
                std::cout << "Vertex-ID: \t" << v.getID() << std::endl;
                std::cout << "Type: \t" << get_vertex_type(v.getType()) << std::endl;
                std::cout << "Properties: ";
                for (const auto entry : v.getProperties()) {
                    auto value = entry.second;
                    std::cout << "{" << entry.first << ": ";
                    std::visit(PropertyValueVisitor{}, value);
                    std::cout << "}";
                }
            }
    };
}

#endif //MORPHSTORE_VERTICES_CONTAINER_H