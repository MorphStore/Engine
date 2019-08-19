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
 * @brief Edge class which represents a relationship
 * @todo
*/

#ifndef MORPHSTORE_EDGE_H
#define MORPHSTORE_EDGE_H

#include <stdint.h>
#include <utility>
#include <string>

namespace morphstore{

    class Edge{

    protected:
        // Edge characteristics
        uint64_t sourceID, targetID;
        unsigned short int relation;
        std::pair<std::string, std::string> property;

    public:

        // Constructor with parameters
        Edge(uint64_t from, uint64_t to, unsigned short int rel){
            setSourceId(from);
            setTargetId(to);
            setRelation(rel);
        }

        Edge(uint64_t from, uint64_t to, unsigned short int rel, std::pair<std::string, std::string> prop){
            setSourceId(from);
            setTargetId(to);
            setRelation(rel);
            setProperty(prop);
        }

        // --------------- Getter and Setter ---------------

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

        unsigned short getRelation() const {
            return relation;
        }

        void setRelation(unsigned short relation) {
            Edge::relation = relation;
        }

        const std::pair<std::string, std::string> &getProperty() const {
            return property;
        }

        void setProperty(const std::pair<std::string, std::string> &property) {
            Edge::property = property;
        }

        // function for sorting algorithms in the importer:
        // compare the target ids
        bool operator<(const Edge& e) const
        {
            return getTargetId() < e.getTargetId();
        }
    };
}

#endif //MORPHSTORE_EDGE_H
