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
 * @file edges_container.h
 * @brief abstract class for storing edges
 * @todo an EntityContainer abstraction (reduce duplicated code)
 */

#ifndef MORPHSTORE_EDGES_CONTAINER_H
#define MORPHSTORE_EDGES_CONTAINER_H

#include <core/storage/graph/edge/edge.h>
#include <core/storage/graph/property_type.h>

#include <assert.h>
#include <map>
#include <unordered_map>
#include <utility>

namespace morphstore {
    enum class EdgesContainerType { HashMapContainer, VectorArrayContainer };

    class EdgesContainer {
    protected:
        uint64_t expected_edge_count = 0;
        uint64_t current_max_edge_id = 0;

        std::map<unsigned short int, std::string> edge_type_dictionary;

        // TODO: try other property storage formats than per vertex .. (triple-store or per property)
        std::unordered_map<uint64_t, std::unordered_map<std::string, property_type>> edge_properties;

        std::string get_edge_type(unsigned short int type) const {
            if (edge_type_dictionary.find(type) != edge_type_dictionary.end()) {
                return edge_type_dictionary.at(type);
            } else {
                return "No Matching of type-number in the database! For type " + std::to_string(type);
            }
        }

        uint64_t get_next_edge_id() { return current_max_edge_id++; }

    public:
        virtual std::string container_description() const = 0;
        virtual void insert_edge(EdgeWithId e) = 0;
        virtual EdgeWithId get_edge(uint64_t id) = 0;
        virtual bool exists_edge(const uint64_t id) const = 0;
        virtual uint64_t edge_count() const = 0;

        virtual void allocate(uint64_t expected_edges) {
            edge_properties.reserve(expected_edges);
            expected_edge_count += expected_edges;
        }

        uint64_t add_edge(Edge edge) {
            auto id = get_next_edge_id();
            insert_edge(EdgeWithId(id, edge));
            return id;
        }

        uint64_t add_edge(EdgeWithProperties edge) {
            auto id = add_edge(edge.getEdge());

            if (auto properties = edge.getProperties(); !properties.empty()) {
                edge_properties[id] = properties;
            }

            return id;
        }

        bool has_properties(uint64_t id) { return edge_properties.find(id) != edge_properties.end(); }

        void add_property_to_edge(uint64_t id, const std::pair<std::string, property_type> property) {
            assert(exists_edge(id));
            edge_properties[id].insert(property);
        };

        void set_edge_properties(uint64_t id, const std::unordered_map<std::string, property_type> properties) {
            assert(exists_edge(id));

            if (has_properties(id)) {
                std::cout << "Overwritting existing properties for :";
                print_edge_by_id(id);
                std::cout << std::endl;
            }

            edge_properties[id] = properties;
        };

        void set_edge_type_dictionary(const std::map<unsigned short, std::string> &types) {
            assert(types.size() != 0);
            this->edge_type_dictionary = types;
        }

        const EdgeWithIdAndProperties get_edge_with_properties(uint64_t id) {
            assert(exists_edge(id));
            return EdgeWithIdAndProperties(get_edge(id), edge_properties[id]);
        }

        uint64_t edges_with_properties_count() { return edge_properties.size(); }

        virtual std::pair<size_t, size_t> get_size() const {
            size_t data_size = 0;
            size_t index_size = 0;

            // lookup type dicts
            index_size += 2 * sizeof(std::map<unsigned short int, std::string>);
            for (auto &type_mapping : edge_type_dictionary) {
                index_size += sizeof(unsigned short int);
                index_size += sizeof(char) * (type_mapping.second.length());
            }

            // edge-properties:
            index_size += sizeof(std::unordered_map<uint64_t, std::unordered_map<std::string, std::string>>);
            for (const auto &property_mapping : edge_properties) {
                index_size += sizeof(uint64_t) + sizeof(std::unordered_map<std::string, std::string>);
                for (const auto &property : property_mapping.second) {
                    data_size += sizeof(char) * property.first.length() + sizeof(property.second);
                }
            }

            return {index_size, data_size};
        }

        void print_type_dict() {
            std::cout << "EdgeType-Dict: " << std::endl;
            for (auto const &entry : edge_type_dictionary) {
                std::cout << entry.first << " -> " << entry.second << std::endl;
            }
        }

        void print_edge_by_id(const uint64_t id) {
            std::cout << "-------------- Edge ID: " << id << " --------------" << std::endl;
            auto e = get_edge_with_properties(id);
            std::cout << e.getEdge().to_string() << std::endl;
            std::cout << "Type: " << this->get_edge_type(e.getEdge().getType()) << std::endl;
            std::cout << "Properties: ";
            for (const auto entry : e.getProperties()) {
                auto value = entry.second;
                std::cout << "{" << entry.first << ": ";
                std::visit(PropertyValueVisitor{}, value);
                std::cout << "}";
            }
            std::cout << std::endl;
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_EDGES_CONTAINER_H