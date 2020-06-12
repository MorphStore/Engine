/**********************************************************************************************
 * Copyright (C) 2019-2020 by MorphStore-Team                                                      *
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
 * @file top_down_bfs.h
 * @brief top down BFS implementation to traverse graph
 * @todo implement vectorized BFS (AVX2, AVX-512)
 */

#ifndef MORPHSTORE_TOP_DOWN_BFS
#define MORPHSTORE_TOP_DOWN_BFS

#include <core/operators/graph/degree_measurement.h>
#include <core/storage/graph/graph.h>

#include <chrono>

namespace morphstore {
    class BFS {

    public:
        // actual BFS algorithm: takes the start-node id and returns the number of explored vertices
        static uint64_t compute(std::shared_ptr<Graph> graph, uint64_t startVertex) {
            std::vector<uint64_t> frontier;
            std::vector<uint64_t> next;
            std::vector<bool> visited(graph->getVertexCount(), false);
            uint64_t exploredVertices = 0;

            frontier.push_back(startVertex);
            visited[startVertex] = true;

            while (!frontier.empty()) {
                // Loop through current layer of vertices in the frontier
                for (uint64_t i = 0; i < frontier.size(); ++i) {
                    uint64_t currentVertex = frontier[i];
                    // get list of a vertex's adjacency
                    std::vector<uint64_t> neighbors = graph->get_neighbors_ids(currentVertex);

                    // Loop through all of neighbors of current vertex
                    for (uint64_t j = 0; j < neighbors.size(); ++j) {
                        // check if neighbor has been visited, if not -> put into frontier and mark as visit = true
                        if (!visited[neighbors[j]]) {
                            next.push_back(neighbors[j]);
                            visited[neighbors[j]] = true;
                            ++exploredVertices;
                        }
                    }
                }
                // swap frontier with next
                frontier.swap(next);
                // clear next: swap with an empty container is faster
                std::vector<uint64_t>().swap(next);
            }
            return exploredVertices;
        }

        // ------------------------------------------ Measurement stuff ------------------------------------------

        // function that measures the number of explored vertices and time in ms:
        // results are written into a file; cycle determines the ith vertex from list
        static void do_measurements(std::shared_ptr<Graph> graph, uint64_t cycle, std::string pathToFile) {
            // list of measurement candidates: the parameter means the ith vertex in total
            std::vector<uint64_t> candidates = get_list_of_every_ith_vertex(graph, cycle);

            // Intermediate data structure: (explored vertices, time in ms)
            std::vector<std::pair<uint64_t, uint64_t>> results;
            results.reserve(candidates.size());

            for (uint64_t i = 0; i < candidates.size(); ++i) {
                // start measuring bfs time:
                auto startBFSTime = std::chrono::high_resolution_clock::now();

                uint64_t exploredVertices = compute(graph, candidates[i]);

                auto finishBFSTime = std::chrono::high_resolution_clock::now(); // For measuring the execution time
                auto elapsedBFSTime =
                    std::chrono::duration_cast<std::chrono::milliseconds>(finishBFSTime - startBFSTime).count();

                // write to intermediate array:
                results.push_back({exploredVertices, elapsedBFSTime});
            }

            // WRITE INTERMEDIATES TO FILE:
            std::ofstream fs;
            std::stringstream ss;
            std::string filename = pathToFile;
            // open file for writing and delete existing stuff:
            fs.open(filename, std::fstream::out | std::ofstream::trunc);

            ss << "explored vertices | time in ms \n";

            for (uint64_t j = 0; j < results.size(); j++) {
                ss << results[j].first << "," << results[j].second << "\n";
            }
            fs << ss.str();

            fs.close();
        }

        // function which returns a list of every ith vertex which is sorted by degree DESC
        static std::vector<uint64_t> get_list_of_every_ith_vertex(std::shared_ptr<Graph> graph, uint64_t cycle) {
            std::vector<uint64_t> measurementCandidates;
            std::vector<std::pair<uint64_t, uint64_t>> totalListOfVertices =
                DegreeMeasurement::get_list_of_degree_DESC(graph);
            for (uint64_t i = 0; i < totalListOfVertices.size(); i = i + cycle) {
                measurementCandidates.push_back(totalListOfVertices[i].first);
            }
            return measurementCandidates;
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_TOP_DOWN_BFS
