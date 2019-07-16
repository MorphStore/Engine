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
 * @file graph_abstract.h
 * @brief this abstract class is for the ldbc importer for polymorphism of different storage formats (pointer to derived classes)
 * @todo add all used functions of graphs in ldbc importer class
*/

#ifndef MORPHSTORE_GRAPH_ABSTRACT_H
#define MORPHSTORE_GRAPH_ABSTRACT_H

#include <string>

namespace morphstore{

    class Graph{
    public:
        //virtual ~Graph();
        virtual std::string getStorageFormat() = 0;
        virtual size_t get_size_of_graph() = 0;
        
        // AdjacecenyList functions for ldbc-importer:
        virtual void add_vertex() = 0;
        virtual void add_edge(const uint64_t sourceID, const uint64_t targetID, const std::string& rel) = 0;
        virtual void add_edge_with_property(uint64_t sourceID, uint64_t targetID, const std::string& rel, const std::pair<std::string, std::string>& property) = 0;
        virtual uint64_t add_vertex_with_properties(const std::unordered_map<std::string, std::string>& props ) = 0;
        virtual void add_entity_to_vertex(const uint64_t id, const std::string& entity) = 0;
        virtual void add_property_to_vertex(uint64_t id, const std::pair<std::string, const std::string>& property) = 0;

    };
}

#endif //MORPHSTORE_GRAPH_ABSTRACT_H
