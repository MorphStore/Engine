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
        //std::vector<bool> visited = { false };
        //std::vector<uint64_t> layer;
        // Create a queue for the nodes we visit.
    
     public:

        // constructor with smart pointer to graph as parameter
        BFS(std::unique_ptr<morphstore::Graph>& g) : graph(std::move(g)){
            graphSize = graph->getNumberVertices();
            //visited.resize(graphSize);
            //layer.resize(graphSize);
        }

        uint64_t get_graph_size(){
            return graphSize;
        }

        // actual BFS (naive) algorithm: takes the start-node id and returns the number of explored vertices
        uint64_t do_BFS(uint64_t startVertex){
            std::vector<uint64_t> frontier;
            std::vector<uint64_t> next;
	    std::vector<bool> visited(graphSize, false);
	   
	    // debug: 
	    //int layer = 0;
	    //int layerVertices = 0;

            // set every entry in visited array back to { false }
            //clear_visited_array();

            uint64_t exploredVertices = 0;

            frontier.push_back(startVertex);
            visited[startVertex] = true;

            //layer[startVertex] = 0;

            while(!frontier.empty()){
                // Loop through current layer of vertices in the frontier
                for(uint64_t i = 0; i < frontier.size(); ++i){
                    uint64_t currentVertex = frontier[i];
		    std::vector<uint64_t> neighbors = graph->get_neighbors_ids(currentVertex);	
                    // Loop through all of neighbors of current vertex
                    for(uint64_t j = 0; j < neighbors.size(); ++j){
			// check if neighbor has been visited, if not -> put into queue and mark as visit = true
                        if(!visited[neighbors[j]]){
                            next.push_back(neighbors[j]);
                            //layer[neighbor] = layer[currentVertex] +1;
                            visited[neighbors[j]] = true;
                            ++exploredVertices;
		            //++layerVertices;
                        }
                    }
                }
		//++layer;
		//std::cout << "Explored layer " << layer << " -> " << layerVertices  << std::endl;
        	//layerVertices = 0;
	        // swap frontier with next
                frontier.swap(next);
                // clear next: swap with an empty container is much faster
                std::vector<uint64_t>().swap(next);
		
                //std::cout << "Vertex with ID " << currentVertex << "\t @ Layer " << layer[currentVertex] << std::endl;
            }
            return exploredVertices;
  	}

	// function that measures the number of explored vertices and TIME:
        // results are written into a file
        // parameter cycle means the ith vertex (modulo)
        void do_measurements(uint64_t cycle, std::string pathToFile){

            // list of measurement candidates: the parameter means the ith vertex in total
            std::vector<uint64_t> candidates = get_list_of_every_ith_vertex(cycle);

            // Intermediate data structure:
            // size = candidatesVector size*2, because we sequentially store both results (exploredVertices, needed Time) for every vertex
            std::vector<std::pair<uint64_t, uint64_t>> results;
	    results.reserve(candidates.size());


            for(uint64_t i = 0; i < candidates.size(); ++i){
                // start measuring bfs time:
                auto startBFSTime = std::chrono::high_resolution_clock::now();

                uint64_t exploredVertices = do_BFS(candidates[i]);

                auto finishBFSTime = std::chrono::high_resolution_clock::now(); // For measuring the execution time
                auto elapsedBFSTime = std::chrono::duration_cast< std::chrono::milliseconds >( finishBFSTime - startBFSTime ).count();

                // write to intermediate array:
                results.push_back({exploredVertices, elapsedBFSTime});
            }

            // WRITE INTERMEDIATES TO FILE:
            std::ofstream fs;
            std::stringstream ss;
            std::string filename = pathToFile;
            // open file for writing and delete existing stuff:
            fs.open(filename, std::fstream::out | std::ofstream::trunc);

            for(uint64_t j = 0; j < results.size(); ++j){
                ss << results[j].first << "," << results[j].second << "\n";
                ++j;
            }
            fs << ss.str() ;

            fs.close();
        }

	// function which returns a list of every ith vertex which is sorted by degree DESC
        std::vector< uint64_t > get_list_of_every_ith_vertex(uint64_t cycle){
            std::vector< uint64_t > measurementCandidates;
            std::vector< std::pair<uint64_t, uint64_t> > totalListOfVertices = graph->get_list_of_degree_DESC();
            for(uint64_t i = 0; i < totalListOfVertices.size(); i = i + cycle){
                measurementCandidates.push_back(totalListOfVertices[i].first);
            }
            return measurementCandidates;
        }

    };
}

#endif //MORPHSTORE_BFS_NAIVE_H

