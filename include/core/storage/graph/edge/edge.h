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

#include "../property_type.h"

#include <stdint.h>
#include <utility>
#include <string>
#include <iostream>
#include <unordered_map>
#include <memory>

namespace morphstore{

    class Edge{

    protected:
        // Edge characteristics
        uint64_t sourceID, targetID, id;
        unsigned short int type;

        // delete flag
        // TODO put as a std::bitset in vectorarray_container
        bool valid = false;

        uint64_t getNextEdgeId() const {
            // Todo: enable resetting maxEdgeId 
            // Ideal would be to pull id gen to graph.h but this requires rewriting Ldbc importer to use (edge property setting depends on it)
            static uint64_t currentMaxEdgeId = 0;
            return currentMaxEdgeId++;
        }

    public:
        // default constr. needed for EdgeWithProperties constructor
        Edge(){}

        Edge(uint64_t sourceId, uint64_t targetId, unsigned short int type)
            : Edge(getNextEdgeId(), sourceId, targetId, type) {}

        Edge(uint64_t id, uint64_t sourceId, uint64_t targetId, unsigned short int type){
            this->sourceID = sourceId;
            this->targetID = targetId;
            this->type = type;
            this->id = id;
            this->valid = true;
        }

        // this is needed for csr when doing edge_array[offset] = edge...
        Edge& operator= (const Edge &edge){
            // self-assignment guard
            if (this == &edge)
                return *this;

            // do the copy
            this->sourceID = edge.sourceID;
            this->targetID = edge.targetID;
            this->type = edge.type;
            this->id = edge.id;
            this->valid = edge.valid;

            // return the existing object so we can chain this operator
            return *this;
        }

        // --------------- Getter and Setter ---------------

        uint64_t getId() const {
            return id;
        }

        uint64_t getSourceId() const {
            return sourceID;
        }

        uint64_t getTargetId() const {
            return targetID;
        }

        unsigned short getType() const {
            return type;
        }

        bool isValid() const {
            return valid;
        }

        // function for sorting algorithms in the ldbc-importer:
        // compare target-ids and return if it's "lower" (we need the sorting for the CSR)
        bool operator<(const Edge& e) const{
            return getTargetId() < e.getTargetId();
        }

        // get size of edge object in bytes:
        static size_t size_in_bytes() {
            size_t size = 0;
            size += sizeof(uint64_t) * 3; // id, source- and target-id
            size += sizeof(unsigned short int); // type
            size += sizeof(bool); // valid flag
            return size;
        }

        std::string to_string() const {
            return "(id:" + std::to_string(this->id) + " ," 
            + std::to_string(this->sourceID) + "->" + std::to_string(this->targetID) + " ," 
            + "valid: " + std::to_string(this->valid) + ")";
        }
    };

    class EdgeWithProperties {
        private:
            Edge edge;
            std::unordered_map<std::string, property_type> properties;
        public:
            EdgeWithProperties(Edge edge, const std::unordered_map<std::string, property_type> properties) {
                this->edge = edge;
                this->properties = properties;
            }

            Edge getEdge() {
                return edge;
            }

            std::unordered_map<std::string, property_type> getProperties() {
                return properties;
            }
    };
}

#endif //MORPHSTORE_EDGE_H
