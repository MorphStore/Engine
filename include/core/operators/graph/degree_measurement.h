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
 * @file degree_measurement.h
 * @brief computing a degree distribution of a given graph
 * @todo multi-threaded impl? ; add tests
 */

#ifndef MORPHSTORE_DEGREE_MEASUREMENT
#define MORPHSTORE_DEGREE_MEASUREMENT

#include <core/storage/graph/graph.h>

#include <chrono>

namespace morphstore {

    class DegreeMeasurement {

    public:
        // function to return a list of pair < vertex id, degree > DESC:
        static std::vector<std::pair<uint64_t, uint64_t>> get_list_of_degree_DESC(std::shared_ptr<Graph> &graph) {
            std::vector<std::pair<uint64_t, uint64_t>> vertexDegreeList;
            auto vertex_count = graph->getVertexCount();
            vertexDegreeList.reserve(vertex_count);

            // fill the vector with every vertex key and his degree
            for (uint64_t i = 0; i < vertex_count; ++i) {
#if DEBUG
                if (i % 10000 == 0) {
                    std::cout << "Degree-List - Current Progress" << i << "/" << vertex_count << std::endl;
                }
#endif
                vertexDegreeList.push_back({i, graph->get_out_degree(i)});
            }

            // sort the vector on degree DESC
            std::sort(vertexDegreeList.begin(), vertexDegreeList.end(),
                      [](const std::pair<uint64_t, uint64_t> &left, const std::pair<uint64_t, uint64_t> &right) {
                          return left.second > right.second;
                      });

            return vertexDegreeList;
        }

        // function to measure graph characteristics (degree and count) and write the result to a given file:
        static void measure_degree_count(std::shared_ptr<Graph> graph, std::string filePath) {
            std::vector<std::pair<uint64_t, uint64_t>> verticesDegree = get_list_of_degree_DESC(graph);
            // unordered map for mapping degree to count:
            std::unordered_map<uint64_t, uint64_t> results;

            for (uint64_t i = 0; i < verticesDegree.size(); ++i) {
                // increment count in results for a given degree:
                results[verticesDegree[i].second]++;
            }

            // write to file:
            std::ofstream fs;
            std::stringstream ss;
            // open file for writing and delete existing stuff:
            fs.open(filePath, std::fstream::out | std::ofstream::trunc);

            for (auto const &m : results) {
                ss << m.first << "," << m.second << "\n";
            }
            fs << ss.str();
            fs.close();
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_DEGREE_MEASUREMENT
