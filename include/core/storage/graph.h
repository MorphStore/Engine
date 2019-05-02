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
 * @file graph.h
 * @brief Graph storage format -> adjacency Lists
 * @todo Add property structure to Vertex and Edge
 */

#ifndef MORPHSTORE_GRAPH_H
#define MORPHSTORE_GRAPH_H

#include <unordered_map>
#include <vector>

namespace graph{

    struct Vertex;
    struct Edge;

    struct Vertex{
        uint64_t id;
        uint64_t ldbc_id;
        int entity;
        vector<Edge> adjList;
    };

    struct Edge{
        Vertex* target;
        int relation;
    };

    struct graph{
        unordered_map<unint64_t, Vertex> vertices;
        void addVertex();
        void addEdge;
    };

}

#endif //MORPHSTORE_GRAPH_H
