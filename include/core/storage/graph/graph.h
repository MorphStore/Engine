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
 * @brief base graph class for any storage format --> CSR,ADJ (allowing multi-graphs)
 * @todo
 */

#ifndef MORPHSTORE_GRAPH_H
#define MORPHSTORE_GRAPH_H

#include "edge/edge.h"
#include "edge/edges_hashmap_container.h"
#include "edge/edges_vectorarray_container.h"
#include "property_type.h"
#include "vertex/vertex.h"
#include "vertex/vertices_hashmap_container.h"
#include "vertex/vertices_vectorarray_container.h"
#include <core/storage/graph/graph_compr_format.h>

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace morphstore {

    class Graph {

    protected:
        // TODO: actually just needed for CSR format (could be moved)
        uint64_t expectedVertexCount;
        uint64_t expectedEdgeCount;

        std::unique_ptr<VerticesContainer> vertices;
        std::unique_ptr<EdgesContainer> edges;

        // graph format specific (CSR and Adj only differ in their graph topology representation)
        virtual void add_to_vertex_edges_mapping(uint64_t sourceID, const std::vector<uint64_t> edge_ids) = 0;

    public:
        Graph(EdgesContainerType edges_container_type)
            : Graph(VerticesContainerType::VectorArrayContainer, edges_container_type) {}

        Graph(VerticesContainerType vertices_container_type = VerticesContainerType::VectorArrayContainer,
              EdgesContainerType edges_container_type = EdgesContainerType::VectorArrayContainer) {
            // could be encapsulated in a VerticesContainer builder
            switch (vertices_container_type) {
            case VerticesContainerType::VectorArrayContainer:
                vertices = std::make_unique<VerticesVectorArrayContainer>();
                break;
            case VerticesContainerType::HashMapContainer:
                vertices = std::make_unique<VerticesHashMapContainer>();
                break;
            }

            // could be encapsulated in a EdgesContainer builder
            switch (edges_container_type) {
            case EdgesContainerType::VectorArrayContainer:
                edges = std::make_unique<EdgesVectorArrayContainer>();
                break;
            case EdgesContainerType::HashMapContainer:
                edges = std::make_unique<EdgesHashMapContainer>();
                break;
            }
        }

        // human-readable form of the container (f.i. for benchmark)
        std::string vertices_container_description() { return vertices->container_description(); }
        
        // human-readable form of the container (f.i. for benchmark)
        std::string edges_container_description() { return edges->container_description(); }

        // -------------------- Setters & Getters --------------------

        // each vertex has a type represented by a number (in Neo4j terms this would be a node label)
        // this provides the semantics behind that number
        void set_vertex_type_dictionary(const std::map<unsigned short, std::string> &types) {
            assert(types.size() != 0);
            this->vertices->set_vertex_type_dictionary(types);
        }

        // each edge has a type represented by a number (in Neo4j terms this would be a relationship type)
        // this provides the semantics behind that number
        void setEdgeTypeDictionary(const std::map<unsigned short, std::string> &types) {
            assert(types.size() != 0);
            this->edges->set_edge_type_dictionary(types);
        }

        // expected count provided by allocate_graph_structure
        uint64_t getExpectedVertexCount() const { return expectedVertexCount; }

        // count of actually stored vertices
        uint64_t getVertexCount() const { return vertices->vertex_count(); }

        // expected count provided by allocate_graph_structure
        uint64_t getExpectedEdgeCount() const { return expectedEdgeCount; }

        // count of actually stored edges
        uint64_t getEdgeCount() const { return edges->edge_count(); }

        uint64_t add_vertex(const unsigned short int type = 0,
                            const std::unordered_map<std::string, property_type> props = {}) {
            return vertices->add_vertex(type, props);
        };

        VertexWithProperties get_vertex(uint64_t id) { return vertices->get_vertex_with_properties(id); }

        EdgeWithIdAndProperties get_edge(uint64_t id) { return edges->get_edge_with_properties(id); }

        void add_property_to_vertex(uint64_t id, const std::pair<std::string, property_type> property) {
            vertices->add_property_to_vertex(id, property);
        };

        // only setting whole edge_properties, as adding an edge property was not needed yet
        void set_edge_properties(uint64_t id, const std::unordered_map<std::string, property_type> properties) {
            edges->set_edge_properties(id, properties);
        };

        // human-readable form of the graph storage format
        virtual std::string get_storage_format() const = 0;
        virtual std::string get_compression_format() const = 0;
        virtual uint64_t add_edge(uint64_t from, uint64_t to, unsigned short int type) = 0;
        // changing the compression format
        virtual void morph(GraphCompressionFormat target_format) = 0;
        // outgoing, as they are only indexed in the outgoing direction
        virtual std::vector<uint64_t> get_outgoing_edge_ids(uint64_t id) = 0;
        // get the out_degree of a vertex (size of the adjacency list)
        virtual uint64_t get_out_degree(uint64_t id) = 0;

        // convenience method to returning the target vertex-ids of the outgoing edges 
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) {
            std::vector<uint64_t> targetVertexIds;
            // guess this could be easily parallelized (using std::foreach f.i.)
            for (auto edge_id : get_outgoing_edge_ids(id)) {
                assert(edges->exists_edge(edge_id));
                targetVertexIds.push_back(edges->get_edge(edge_id).getTargetId());
            }

            return targetVertexIds;
        };

        // returning a vector of edge-ids (order based on input edges)
        std::vector<uint64_t> add_edges(uint64_t sourceId, const std::vector<Edge> edges_to_add) {
            std::vector<uint64_t> edge_ids;

            // assertion, which are shared by all graph formats
            if (!vertices->exists_vertex(sourceId)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceId));
            }

            // (multi)-graph specific and storage-format agnostic 
            // changes, if other formats store target ids instead of edge ids (because non multi graphs do not need edge ids) 
            for (auto edge : edges_to_add) {
                if (!vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
                edge_ids.push_back(edges->add_edge(edge));
            }

            add_to_vertex_edges_mapping(sourceId, edge_ids);

            return edge_ids;
        };

        // looks very similar to above but for ! EdgeWithProperties !
        // extra method, as runtime polymorphism seemed ugly in C++ here (but very likely there is a better way for this)
        std::vector<uint64_t> add_edges(uint64_t sourceId, const std::vector<EdgeWithProperties> edges_to_add) {
            std::vector<uint64_t> edge_ids;

            if (!vertices->exists_vertex(sourceId)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceId));
            }

            for (auto edge_with_props : edges_to_add) {
                if (auto edge = edge_with_props.getEdge(); !vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
                // this calls a different methods on the edges-container
                edge_ids.push_back(edges->add_edge(edge_with_props));
            }

            add_to_vertex_edges_mapping(sourceId, edge_ids);

            return edge_ids;
        };

        // memory estimation 
        // returns a pair of index-size, data-size
        virtual std::pair<size_t, size_t> get_size_of_graph() const {
            // including vertices + its properties + its type dict
            auto [index_size, data_size] = vertices->get_size();

            // including edges + its properties + its type dict
            auto edges_size = edges->get_size();
            index_size += edges_size.first;
            data_size += edges_size.second;

            return {index_size, data_size};
        };


        // mainly needed to allocate CSR columns
        // also containers can reserve expected size
        virtual void allocate_graph_structure(uint64_t expected_vertices, uint64_t expected_edges) {
            this->expectedVertexCount = expected_vertices;
            this->expectedEdgeCount = expected_edges;

            vertices->allocate(expected_vertices);
            edges->allocate(expected_edges);
        };

        // -------------------- debugging functions --------------------

        void print_neighbors_of_vertex(uint64_t id) {
            std::cout << std::endl << "Neighbours for Vertex with id " << id << std::endl;
            auto edge_ids = get_outgoing_edge_ids(id);

            if (edge_ids.size() == 0) {
                std::cout << "  No outgoing edges for vertex with id: " << id << std::endl;
            } else {
                for (const auto edge_id : edge_ids) {
                    print_edge_by_id(edge_id);
                }
            }
        }

        // basic statistics to be extended by graph formats
        virtual void statistics() {
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << getVertexCount() << std::endl;
            std::cout << "Number of vertices with properties:" << vertices->vertices_with_properties_count()
                      << std::endl;
            std::cout << "Number of edges: " << getEdgeCount() << std::endl;
            std::cout << "Number of edges with properties:" << edges->edges_with_properties_count() << std::endl;
            std::cout << "Compression Format:" << get_compression_format() << std::endl;
        }

        void print_vertex_by_id(uint64_t id) {
            vertices->print_vertex_by_id(id);
            std::cout << "\n";
            std::cout << "#Edges: " << this->get_out_degree(id);
            std::cout << "\n";
            std::cout << "-----------------------------------------------" << std::endl;
        }

        void print_edge_by_id(uint64_t id) { edges->print_edge_by_id(id); }

        void print_type_dicts() {
            vertices->print_type_dict();
            edges->print_type_dict();
        }
    };

} // namespace morphstore

#endif // MORPHSTORE_GRAPH_H
