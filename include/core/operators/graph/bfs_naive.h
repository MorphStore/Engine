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
 * @todo implement vectorized BFS (AVX2, AVX-512)
 */

#ifndef MORPHSTORE_BFS_NAIVE_H
#define MORPHSTORE_BFS_NAIVE_H

#include "../../storage/graph/graph.h"

#include <queue>
#include <chrono>

namespace morphstore{

    class BFS{

    private:
        std::unique_ptr<morphstore::Graph> graph;
        uint64_t graphSize;
        // Create a "visited" array (true or false) to keep track of if we visited a vertex.
        std::vector<bool> visited = { false };
        //std::vector<uint64_t> layer;
        // Create a queue for the nodes we visit.
        std::queue<uint64_t> queue;

    public:

        // constructor with smart pointer to graph as parameter
        BFS(std::unique_ptr<morphstore::Graph>& g) : graph(std::move(g)){
            graphSize = graph->getNumberVertices();
            visited.resize(graphSize);
            //layer.resize(graphSize);
        }

        uint64_t get_graph_size(){
            return graphSize;
        }

        // actual BFS (naive) algorithm: takes the start-node id and returns the number of explored vertices
        uint64_t doBFS(uint64_t startVertex){

            uint64_t exploredVertices = 0;

            queue.push(startVertex);
            visited[startVertex] = true;

            //layer[startVertex] = 0;

            while(!queue.empty()){
                uint64_t currentVertex = queue.front();
                queue.pop();

                //std::cout << "Vertex with ID " << currentVertex << "\t @ Layer " << layer[currentVertex] << std::endl;

                // get neighbors of current vertex
                std::vector<uint64_t> neighbors = graph->get_neighbors_ids(currentVertex);

                // Loop through all of neighbors of current vertex
                for(uint64_t i = 0; i < neighbors.size(); ++i){
                    uint64_t neighbor = neighbors[i];

                    // check if neighbor has been visited, if not -> put into queue and mark as visit = true
                    if(!visited[neighbor]){
                        queue.push(neighbor);
                        //layer[neighbor] = layer[currentVertex] +1;
                        visited[neighbor] = true;
                        ++exploredVertices;
                    }
                }
            }
            return exploredVertices;
        }

        // this function sets every cell to false in visited array
        void clear_visited_array(){
            std::fill(visited.begin(), visited.end(), false);
        }

        // function that measures for every vertex the number and time of explored vertices in BFS
        // writes results to local file for further analysis
        void do_measurements(){

            std::ofstream fs;
            std::string filename = "/home/tim/Documents/TUD/(8) Informatik SS 2019/MorphStore/bfs_measurements.csv";
            // open file for writing and delete existing stuff:
            fs.open(filename, std::fstream::out | std::ofstream::trunc);

            for(uint64_t i = 0; i < graphSize; ++i){
                // start measuring bfs time:
                auto startBFSTime = std::chrono::high_resolution_clock::now();

                uint64_t exploredVertices = doBFS(i);

                auto finishBFSTime = std::chrono::high_resolution_clock::now(); // For measuring the execution time
                auto elapsedBFSTime = std::chrono::duration_cast< std::chrono::milliseconds >( finishBFSTime - startBFSTime ).count();

                // set every entry in visited array back to { false }
                clear_visited_array();

                // write to file
                fs << exploredVertices << "," << elapsedBFSTime << "\n";
            }
            fs.close();
        }

    };
}

#endif //MORPHSTORE_BFS_NAIVE_H
