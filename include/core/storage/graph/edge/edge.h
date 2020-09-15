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
 * @file edge.h
 * @brief Edge class which represents an edge object between two vertices
 * @todo
 */

#ifndef MORPHSTORE_EDGE_H
#define MORPHSTORE_EDGE_H

#include <core/storage/graph/property_type.h>

#include <iostream>
#include <memory>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>

namespace morphstore {

    // for loading a graph
    class Edge {

    protected:
        // Edge characteristics
        uint64_t sourceId, targetId;
        unsigned short int type;

    public:
        Edge() {}

        virtual ~Edge() = default;

        Edge(uint64_t sourceId, uint64_t targetId, unsigned short int type) {
            this->sourceId = sourceId;
            this->targetId = targetId;
            this->type = type;
        }

        // --------------- Getter and Setter ---------------

        uint64_t getSourceId() const { return sourceId; }

        uint64_t getTargetId() const { return targetId; }

        unsigned short getType() const { return type; }

        // function for sorting algorithms in the ldbc-importer:
        // compare target-ids and return if it's "lower" (we need the sorting for the CSR)
        bool operator<(const Edge &e) const { return getTargetId() < e.getTargetId(); }

        // get size of edge object in bytes:
        static size_t size_in_bytes() {
            size_t size = 0;
            size += sizeof(uint64_t) * 2;       // source- and target-id
            size += sizeof(unsigned short int); // type
            return size;
        }

        virtual std::string to_string() const {
            return "(" + std::to_string(this->sourceId) + "->" + std::to_string(this->targetId) + ")";
        }
    };

    // for internal usage (inside the edges-container)
    class EdgeWithId : public Edge {
    private:
        uint64_t id;

        // delete flag
        // TODO: put as a std::bitset in vectorarray_container (as hashmap-container does not need the valid flag)
        bool valid = false;

    public:
        // default constr. needed for EdgeWithProperties constructor
        EdgeWithId() {}

        EdgeWithId(uint64_t id, uint64_t sourceId, uint64_t targetId, unsigned short int type)
            : Edge(sourceId, targetId, type) {
            this->id = id;
            this->valid = true;
        }

        EdgeWithId(uint64_t id, Edge edge) : Edge(edge.getSourceId(), edge.getTargetId(), edge.getType()) {
            this->id = id;
            this->valid = true;
        }

        uint64_t getId() const { return id; }

        bool isValid() const { return valid; }

        // this is needed for edges_container when doing edges[id] = edge
        EdgeWithId &operator=(const EdgeWithId &edge) {
            // self-assignment guard
            if (this == &edge)
                return *this;

            // do the copy
            this->sourceId = edge.getSourceId();
            this->targetId = edge.getTargetId();
            this->type = edge.getType();
            this->id = edge.getId();
            this->valid = edge.isValid();

            // return the existing object so we can chain this operator
            return *this;
        }

        // edge size + id and valid flag
        static size_t size_in_bytes() { return Edge::size_in_bytes() + sizeof(uint64_t) + sizeof(bool); }

        std::string to_string() const override {
            return "(id:" + std::to_string(this->id) + " ," + "valid: " + std::to_string(this->valid) +
                   Edge::to_string() + ")";
        }
    };

    // for loading
    class EdgeWithProperties {
    private:
        std::unordered_map<std::string, property_type> properties;
        // not using inheritance as vector<Edge> elements could not get cast to EdgeWithProperties
        Edge edge;

    public:
        EdgeWithProperties(uint64_t sourceId, uint64_t targetId, unsigned short int type,
                           const std::unordered_map<std::string, property_type> properties) {
            this->edge = Edge(sourceId, targetId, type);
            this->properties = properties;
        }

        EdgeWithProperties(uint64_t sourceId, uint64_t targetId, unsigned short int type) {
            this->edge = Edge(sourceId, targetId, type);
        }

        Edge getEdge() const { return edge; }

        std::unordered_map<std::string, property_type> getProperties() { return properties; }

        bool operator<(const EdgeWithProperties &e) const { return edge.getTargetId() < e.getEdge().getTargetId(); }
    };

    // for returning an edge to the user
    class EdgeWithIdAndProperties {
    private:
        std::unordered_map<std::string, property_type> properties;
        EdgeWithId edge;

    public:
        EdgeWithIdAndProperties(EdgeWithId edge, const std::unordered_map<std::string, property_type> properties) {
            this->edge = edge;
            this->properties = properties;
        }
        EdgeWithId getEdge() { return edge; }

        std::unordered_map<std::string, property_type> getProperties() { return properties; }
    };
} // namespace morphstore

#endif // MORPHSTORE_EDGE_H
