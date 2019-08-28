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
 * @file bfs.h
 * @brief naive (simple) BFS implementation to traverse graph of type CSR OR AdjacencyList
 * @todo implement optimized version of BFS -> now just for simplicity
 */

#ifndef MORPHSTORE_BFS_NAIVE_H
#define MORPHSTORE_BFS_NAIVE_H

#include "../../storage/graph/graph.h"

#include <queue>

namespace morphstore{

    class BFS{

    private:
        std::unique_ptr<morphstore::Graph> graph;
        uint64_t graphSize;
        // Create a "visited" array (true or false) to keep track of if we visited a vertex.
        std::vector<bool> visited = { false };
        std::vector<uint64_t> layer;
        // Create a queue for the nodes we visit.
        std::queue<uint64_t> queue;

    public:

        // constructor with smart pointer to graph as parameter
        BFS(std::unique_ptr<morphstore::Graph>& g) : graph(std::move(g)){
            graphSize = graph->getNumberVertices();
            visited.resize(graphSize);
            layer.resize(graphSize);
        }

        void doBFS(uint64_t startVertex){
            //std::cout << "BFS: starting from Vertex " << startVertex << std::endl;

            queue.push(startVertex);
            visited[startVertex] = true;

            layer[startVertex] = 0;

            while(!queue.empty()){
                uint64_t currentVertex = queue.front();
                queue.pop();

                //std::cout << "Vertex with ID " << currentVertex << "\t @ Layer " << layer[currentVertex] << std::endl;

                // get neighbors of current vertex
                std::vector<uint64_t> neighbors = graph->get_neighbors_ids(currentVertex);

                // Loop through all of neighbors of current vertex
                for(uint64_t i = 0; i < neighbors.size(); i++){
                    uint64_t neighbor = neighbors[i];

                    // check if neighbor has been visited, if not -> put into queue and mark as visit = true
                    if(!visited[neighbor]){
                        queue.push(neighbor);
                        layer[neighbor] = layer[currentVertex] +1;
                        visited[neighbor] = true;
                    }
                }
            }

        }

    };
}

#endif //MORPHSTORE_BFS_NAIVE_H
