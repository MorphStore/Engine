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
 * @file csr.h
 * @brief Derived CSR storage format class. Base: graph.h
 * @todo add_edge() functionality is missing -> needs a realloc()-strategy
*/

#ifndef MORPHSTORE_CSR_H
#define MORPHSTORE_CSR_H

#include "../graph.h"
#include "../vertex/csr_vertex.h"

namespace morphstore{

    class CSR: public Graph{

    private:
        /* graph topology: hybrid approach
         * node array: index is vertex-id; array cell contains offset in edge_array
         * edge array: every cell contains pointer to edge object of vertex
         */
        // TODO: construct a graph-topology struct ?
        // TODO: free memory in destructor
        uint64_t* node_array = nullptr;
        Edge* edge_array = nullptr;

    public:

        storageFormat getStorageFormat() const override {
            return csr;
        }

        // this function gets the number of vertices/edges and allocates memory for the vertices-map and the graph topology arrays
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            setNumberVertices(numberVertices);
            setNumberEdges(numberEdges);

            vertices.reserve(numberVertices);

            node_array = (uint64_t*) malloc(numberVertices * sizeof(uint64_t));
            edge_array = (Edge*) malloc(numberEdges * sizeof(Edge));

            // init node array:
            node_array[0] = 0;
        }

        // adding a single vertex (without any properties, etc...)
        void add_vertex() override {
            std::shared_ptr<Vertex> v = std::make_shared<CSRVertex>();
            vertices[v->getID()] = v;
        }

        // adding a vertex with its properties
        int add_vertex_with_properties(const std::unordered_map<std::string, std::string>& props ) override {
            std::shared_ptr<Vertex> v = std::make_shared<CSRVertex>();
            v->setProperties(props);
            vertices[v->getID()] = v;
            return v->getID();
        }

        // TODO: add a single edge in graph arrays -> needs a memory reallocating stragety
        void add_edge(uint64_t from, uint64_t to, unsigned short int rel) override {
            if(exist_id(from) && exist_id(to)){
                std::cout << rel << std::endl;
            }
        }

        // this function fills the graph-topology-arrays sequentially in the order of vertex-ids ASC
        // every vertex id contains a list of neighbors
        void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge>& relations) override {
            uint64_t offset = node_array[sourceID];
            uint64_t nextOffset = offset + relations.size();

            for(const auto & edge : relations){
                edge_array[offset] = edge;
                ++offset;
            }

            // to avoid segfualt:
            if(sourceID < getNumberVertices()-1){
                node_array[sourceID+1] = nextOffset;
            }
        }

        // function to add a single property to vertex
        void add_property_to_vertex(uint64_t id, const std::pair<std::string, const std::string>& property) override {
            if(exist_id(id)){
                vertices[id]->add_property(property);
            }else{
                std::cout << "Vertex with ID " << id << " not found./property_to_vertex" << std::endl;
            }
        }

        // adding entity to vertex
        void add_entity_to_vertex(const uint64_t id, unsigned short int entity) override {
            if(exist_id(id)){
                vertices[id]->setEntity(entity);
            }else{
                std::cout << "Vertex with ID " << id << " not found./entity_to_vertex." << std::endl;
            }
        }

        // get number of edges of vertex with id
        uint64_t get_degree(uint64_t id) override {
            uint64_t offset = node_array[id];
            uint64_t nextOffset = node_array[id+1];
            uint64_t numberEdges = nextOffset - offset;
            return numberEdges;
        }

        void print_neighbors_of_vertex(uint64_t id) override{
            uint64_t offset = node_array[id];
            uint64_t numberEdges = get_degree(id);

            for(uint64_t i = offset; i < offset+numberEdges; ++i){
                std::cout << "Source-ID: " << edge_array[i].getSourceId() << " - Target-ID: " << edge_array[i].getTargetId() << " - Property: { " << edge_array[i].getProperty().first << ": " << edge_array[i].getProperty().second << " }" << " || ";
            }
        }

        // function to return a vector of ids of neighbors for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
            std::vector<uint64_t> neighbors;
            uint64_t offset = node_array[id];
            uint64_t numberEdges = get_degree(id);

            for(uint64_t i = offset; i < offset+numberEdges; ++i){
                neighbors.push_back(edge_array[i].getTargetId());
            }
            return neighbors;
        }

        /* old-calculation of the graph size in bytes
         * size_t get_size_of_graph(){
            size_t size = 0;
            // pointer to arrays:
            size += sizeof(uint64_t*) * 2 + sizeof(unsigned short int*);
            // vertices:
            size += sizeof(uint64_t) * getNumberVertices();
            // edges:
            size += sizeof(uint64_t) * getNumberEdges();
            // val array:
            size += sizeof(unsigned short int) * getNumberEdges();

            // vertex map wth actual data:
            for(std::unordered_map<uint64_t, CSRVertex>::iterator it = vertices.begin(); it != vertices.end(); ++it){
                size += it->second.get_size_of_vertex();
            }

            return size;
        }
        */

    };

}

#endif //MORPHSTORE_CSR_H
