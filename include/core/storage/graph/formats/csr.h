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
 * @todo Edge_value_array should only store edge-ids (not whole objects)
*/

#ifndef MORPHSTORE_CSR_H
#define MORPHSTORE_CSR_H

#include "../graph.h"
#include "../vertex/vertex.h"

namespace morphstore{

    class CSR: public Graph{

    private:
        /* graph topology:
         * offset array: index is vertex-id; array cell contains offset in edgeId array
         * edgeId array: contains edge id
         */
        uint64_t* offset_array = nullptr;
        uint64_t* edgeId_array = nullptr;

    public:

        storageFormat getStorageFormat() const override {
            return csr;
        }

        // this function gets the number of vertices/edges and allocates memory for the vertices-map and the graph topology arrays
        void allocate_graph_structure(uint64_t numberVertices, uint64_t numberEdges) override {
            this->expectedVertexCount = numberVertices;
            this->expectedEdgeCount = numberEdges;

            vertices.reserve(numberVertices);
            edges.reserve(numberEdges);

            offset_array = (uint64_t*) malloc(numberVertices * sizeof(uint64_t));
            edgeId_array = (uint64_t*) malloc(numberEdges * sizeof(uint64_t));

            // init node array:
            offset_array[0] = 0;
        }

        // TODO: add a single edge in graph arrays -> needs a memory reallocating strategy
        void add_edge(uint64_t sourceId, uint64_t targetId, unsigned short int type) override {
            std::cout << "Singe edge addition not yet implemented for CSR" << sourceId << targetId << type;
        }

        // this function fills the graph-topology-arrays sequentially in the order of vertex-ids ASC
        // every vertex id contains a list of its neighbors
        void add_edges(uint64_t sourceID, const std::vector<morphstore::Edge> edgesToAdd) override {
            uint64_t offset = offset_array[sourceID];
            uint64_t nextOffset = offset + edgesToAdd.size();

            // fill the arrays
            for(const auto& edge : edgesToAdd){
                std::shared_ptr<Edge> ePtr = std::make_shared<Edge>(edge);
                edges[ePtr->getId()] = ePtr;
                edgeId_array[offset] = ePtr->getId();
                ++offset;
            }

            // to avoid buffer overflow:
            if(sourceID < getExpectedVertexCount()-1){
                offset_array[sourceID+1] = nextOffset;
            }
        }

        // function to add a single property to vertex
        void add_property_to_vertex(uint64_t id, const std::pair<std::string, std::string> property) override {
            if(exist_vertexId(id)){
                vertices[id]->add_property(property);
            }else{
                std::cout << "Vertex with ID " << id << " not found./property_to_vertex" << std::endl;
            }
        }

        // get number of edges of vertex with id
        uint64_t get_out_degree(uint64_t id) override {
            uint64_t offset = offset_array[id];
            // special case: last vertex id has no next offset
            uint64_t nextOffset;
            if(id == getExpectedVertexCount() -1){
                nextOffset = getExpectedEdgeCount();
            }else{
                nextOffset = offset_array[id+1];
            }

            if(offset == nextOffset) return 0;
            uint64_t degree = nextOffset - offset;
            return degree;
        }

        // for debugging:
        void print_neighbors_of_vertex(uint64_t id) override{
            std::cout << "Neighbours for Vertex with id " << id << std::endl;
            uint64_t offset = offset_array[id];
            uint64_t numberEdges = get_out_degree(id);

            for(uint64_t i = offset; i < offset+numberEdges; ++i){
                uint64_t edgeId = edgeId_array[i];
                std::cout << "Source-ID: " << edges[edgeId]->getSourceId()
                          << " - Target-ID: " << edges[edgeId]->getTargetId()
                          << " Property: { ";
                edges[i]->print_properties();
                std::cout << std::endl
                          << "   }" << std::endl;
            }
        }

        // function to return a vector of ids of neighbors for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) override {
             std::vector<uint64_t> neighbors;
             uint64_t offset = offset_array[id];
             uint64_t numberEdges = get_out_degree(id);

             // avoiding out of bounds ...
             if( offset < getExpectedEdgeCount()){
                 neighbors.insert(neighbors.end(), edgeId_array+offset, edgeId_array+offset+numberEdges);
             }

             return neighbors;
        }

        // get size of storage format:
        std::pair<size_t, size_t> get_size_of_graph() override {
            std::pair<size_t, size_t> index_data_size;
            size_t data_size = 0;
            size_t index_size = 0;
            // TODO: use Graph::get_size_of_graph() for vertices, edges, vertexTypeDictionary and edgeTypeDictionary

            // lookup dicts: entity dict  + relation dict.
            index_size += 2 * sizeof(std::map<unsigned short int, std::string>);
            for(auto& ent : vertexTypeDictionary){
                index_size += sizeof(unsigned short int);
                index_size += sizeof(char)*(ent.second.length());
            }
            for(auto& rel : edgeTypeDictionary){
                index_size += sizeof(unsigned short int);
                index_size += sizeof(char)*(rel.second.length());
            }

            // container for vertices:
            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<morphstore::Vertex>>);
            for(auto& it : vertices){
                index_size += sizeof(uint64_t) + sizeof(std::shared_ptr<morphstore::Vertex>);
                data_size += it.second->get_data_size_of_vertex();
            }
            
            // container for edges:
            index_size += sizeof(std::unordered_map<uint64_t, std::shared_ptr<morphstore::Edge>>);
            for(auto& it : edges){
                // index size of edge: size of id and sizeof pointer 
                index_size += sizeof(uint64_t) + sizeof(std::shared_ptr<morphstore::Edge>);
                // data size:
                data_size += it.second->size_in_bytes();
            }

            // pointer to arrays:
            index_size += sizeof(uint64_t*) * 2 + sizeof(Edge*);
            // edges array values:
            for(uint64_t i = 0; i < getExpectedEdgeCount(); i++){
                index_size += sizeof(uint64_t); // node_array with offsets
            }

            index_data_size = {index_size, data_size};

            return index_data_size;
        }
    };
}
#endif //MORPHSTORE_CSR_H
