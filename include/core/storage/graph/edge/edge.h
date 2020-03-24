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

#include <stdint.h>
#include <utility>
#include <string>
#include <iostream>

namespace morphstore{

    class Edge{

    protected:
        // Edge characteristics
        uint64_t sourceID, targetID, id;
        unsigned short int type;
        // todo: allow map instead of pair
        std::pair<std::string, std::string> property;

        uint64_t getNextEdgeId() const {
            static uint64_t currentMaxEdgeId = 0;
            return currentMaxEdgeId++;
        }

    public:

        // Constructors with parameters
        Edge(uint64_t from, uint64_t to, unsigned short int type){
            setSourceId(from);
            setTargetId(to);
            setType(type);
            this->id = getNextEdgeId();
        }

        Edge(uint64_t from, uint64_t to, unsigned short int type, std::pair<std::string, std::string> prop){
            setSourceId(from);
            setTargetId(to);
            setType(type);
            setProperty(prop);
            this->id = getNextEdgeId();
        }

        // this is needed for csr when doing edge_array[offset] = edge...
        Edge& operator= (const Edge &edge){
            // self-assignment guard
            if (this == &edge)
                return *this;

            // do the copy
            setSourceId(edge.sourceID);
            setTargetId(edge.targetID);
            setType(edge.type);
            setProperty(edge.property);

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

        void setSourceId(uint64_t sourceId) {
            sourceID = sourceId;
        }

        uint64_t getTargetId() const {
            return targetID;
        }

        void setTargetId(uint64_t targetId) {
            targetID = targetId;
        }

        unsigned short getType() const {
            return type;
        }

        void setType(unsigned short type) {
            Edge::type = type;
        }

        const std::pair<std::string, std::string> &getProperty() const {
            return property;
        }

        void setProperty(const std::pair<std::string, std::string> prop) {
            // first check if there is any key value data, otherwise problems with segfaults
            if(prop.first != "" && prop.second != ""){
                Edge::property.first = prop.first;
                Edge::property.second = prop.second;
            }
        }

        // function for sorting algorithms in the ldbc-importer:
        // compare target-ids and return if it's "lower" (we need the sorting for the CSR)
        bool operator<(const Edge& e) const{
            return getTargetId() < e.getTargetId();
        }

        // get size of edge object in bytes:
        size_t size_in_bytes() const{
            size_t size = 0;
            size += sizeof(uint64_t) * 2; // source- and target-id
            size += sizeof(unsigned short int); // relation

            // property:
            size += sizeof(std::pair<std::string, std::string>);
            size += sizeof(char)*(property.first.length() + property.second.length());

            return size;
        }


        // ----------------- DEBUGGING -----------------
        void print_properties() {
            std::cout << "{" << getProperty().first << ": " << getProperty().second << "}";
            std::cout << "\n";
        }
    };
}

#endif //MORPHSTORE_EDGE_H