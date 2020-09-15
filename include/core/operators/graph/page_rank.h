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
 * @file page_rank.h
 * @brief naive page-rank implementation (based on https://en.wikipedia.org/wiki/PageRank)
 * @todo multi-threaded impl? ; add tests; weighted implementation
 */

#ifndef MORPHSTORE_PAGE_RANK
#define MORPHSTORE_PAGE_RANK

#include <core/storage/graph/graph.h>

// for equal with tolerance
#include <algorithm>
// for std::abs
#include <math.h>

namespace morphstore {

    struct PageRankResult {
        // input parameters
        uint64_t max_iterations;
        float damping_factor, tolerance;

        uint64_t ran_iterations = 0;
        // terminated as scores converged?
        bool converged;
        // i-th entry for vertex with id i
        std::vector<float> scores;

        // leaving out the scores
        std::string describe() {
            std::string converged_str = converged ? "True" : "False";
            return "Input-Parameters: { damping-factor: " + std::to_string(damping_factor) +
                   ", max-iterations: " + std::to_string(max_iterations) +
                   ", tolerance: " + std::to_string(tolerance) + "} \n\t\t\t" +
                   "Computed: { converged: " + converged_str + ", ran_iterations: " + std::to_string(ran_iterations) +
                   "}";
        }
    };

    class PageRank {

    public:
        // assuming a consecutive vertex id-space
        static PageRankResult compute(std::shared_ptr<Graph> graph, const uint64_t max_iterations = 20,
                                      const float damping_factor = 0.85, const float tolerance = 0.0001) {
            // init score vector with 1/vertex_count;
            const uint64_t vertex_count = graph->getVertexCount();
            std::vector<float> scores(vertex_count, 1.0 / vertex_count);

            uint64_t iteration;
            bool converged = false;

            for (iteration = 0; iteration < max_iterations; iteration++) {
                // init scores of current iteration
                std::vector<float> new_scores(vertex_count, (1.0 - damping_factor) / vertex_count);

                // loop over all vertices
                for (uint64_t i = 0; i < vertex_count; ++i) {
                    const auto neighbors = graph->get_neighbors_ids(i);

                    // damping_factor * (prev-it-PR(i) / degr(i)) 
                    const auto value_to_propagate = damping_factor * (scores[i] / neighbors.size());

                    // propagate score to its neighbours
                    for (auto neighbor_id : neighbors) {
                        new_scores[neighbor_id] += value_to_propagate;
                    }
                }

                if (std::equal(scores.begin(), scores.end(), new_scores.begin(), new_scores.end(),
                               [tolerance](float score, float other_score) {
                                   return std::abs(score - other_score) < tolerance;
                               })) {
                    converged = true;
                    break;
                }

                scores = new_scores;
            }

            // build result;
            PageRankResult result;
            result.damping_factor = damping_factor;
            result.max_iterations = max_iterations;
            result.tolerance = tolerance;

            result.converged = converged;
            result.ran_iterations = iteration;
            result.scores = scores;

            return result;
        }
    };
} // namespace morphstore

#endif // MORPHSTORE_PAGE_RANK
