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
 * @brief base graph class for any storage format --> CSR,ADJ
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
        GraphCompressionFormat current_compression = GraphCompressionFormat::UNCOMPRESSED;

        // TODO: actually just needed for CSR format (could be moved)
        uint64_t expectedVertexCount;
        uint64_t expectedEdgeCount;

        std::unique_ptr<VerticesContainer> vertices;
        std::unique_ptr<EdgesContainer> edges;

        virtual void add_to_vertex_edges_mapping(uint64_t sourceID, const std::vector<uint64_t> edge_ids) = 0;

    public:
        Graph(EdgesContainerType edges_container_type)
            : Graph(VerticesContainerType::VectorArrayContainer, edges_container_type) {}

        Graph(VerticesContainerType vertices_container_type = VerticesContainerType::VectorArrayContainer,
              EdgesContainerType edges_container_type = EdgesContainerType::VectorArrayContainer) {
            switch (vertices_container_type) {
            case VerticesContainerType::VectorArrayContainer:
                vertices = std::make_unique<VerticesVectorArrayContainer>();
                break;
            case VerticesContainerType::HashMapContainer:
                vertices = std::make_unique<VerticesHashMapContainer>();
                break;
            }

            switch (edges_container_type) {
            case EdgesContainerType::VectorArrayContainer:
                edges = std::make_unique<EdgesVectorArrayContainer>();
                break;
            case EdgesContainerType::HashMapContainer:
                edges = std::make_unique<EdgesHashMapContainer>();
                break;
            }
        }

        std::string vertices_container_description() { return vertices->container_description(); }

        std::string edges_container_description() { return edges->container_description(); }

        // -------------------- Setters & Getters --------------------

        void set_vertex_type_dictionary(const std::map<unsigned short, std::string> &types) {
            assert(types.size() != 0);
            this->vertices->set_vertex_type_dictionary(types);
        }

        void setEdgeTypeDictionary(const std::map<unsigned short, std::string> &types) {
            assert(types.size() != 0);
            this->edges->set_edge_type_dictionary(types);
        }

        uint64_t getExpectedVertexCount() const { return expectedVertexCount; }

        uint64_t getVertexCount() const { return vertices->vertex_count(); }

        uint64_t getExpectedEdgeCount() const { return expectedEdgeCount; }

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

        void set_edge_properties(uint64_t id, const std::unordered_map<std::string, property_type> properties) {
            edges->set_edge_properties(id, properties);
        };

        virtual std::string get_storage_format() const = 0;
        virtual uint64_t add_edge(uint64_t from, uint64_t to, unsigned short int type) = 0;
        virtual void morph(GraphCompressionFormat target_format) = 0;
        virtual std::vector<uint64_t> get_outgoing_edge_ids(uint64_t id) = 0;
        virtual uint64_t get_out_degree(uint64_t id) = 0;

        // function to return a vector of ids of neighbors for BFS alg.
        std::vector<uint64_t> get_neighbors_ids(uint64_t id) {
            std::vector<uint64_t> targetVertexIds;
            for (auto edge_id : get_outgoing_edge_ids(id)) {
                assert(edges->exists_edge(edge_id));
                targetVertexIds.push_back(edges->get_edge(edge_id).getTargetId());
            }

            return targetVertexIds;
        };

        std::vector<uint64_t> add_edges(uint64_t sourceId, const std::vector<Edge> edges_to_add) {
            std::vector<uint64_t> edge_ids;

            if (!vertices->exists_vertex(sourceId)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceId));
            }

            for (auto edge : edges_to_add) {
                if (!vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
                edge_ids.push_back(edges->add_edge(edge));
            }

            add_to_vertex_edges_mapping(sourceId, edge_ids);

            return edge_ids;
        };

        std::vector<uint64_t> add_edges(uint64_t sourceId, const std::vector<EdgeWithProperties> edges_to_add) {
            std::vector<uint64_t> edge_ids;

            if (!vertices->exists_vertex(sourceId)) {
                throw std::runtime_error("Source-id not found " + std::to_string(sourceId));
            }

            for (auto edge_with_props : edges_to_add) {
                if (auto edge = edge_with_props.getEdge(); !vertices->exists_vertex(edge.getTargetId())) {
                    throw std::runtime_error("Target not found  :" + edge.to_string());
                }
                edge_ids.push_back(edges->add_edge(edge_with_props));
            }

            add_to_vertex_edges_mapping(sourceId, edge_ids);

            return edge_ids;
        };

        virtual std::pair<size_t, size_t> get_size_of_graph() const {
            // including vertices + its properties + its type dict
            auto [index_size, data_size] = vertices->get_size();

            // including edges + its properties + its type dict
            auto edges_size = edges->get_size();
            index_size += edges_size.first;
            data_size += edges_size.second;

            return std::make_pair(index_size, data_size);
        };

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

        virtual void statistics() {
            std::cout << "---------------- Statistics ----------------" << std::endl;
            std::cout << "Number of vertices: " << getVertexCount() << std::endl;
            std::cout << "Number of vertices with properties:" << vertices->vertices_with_properties_count()
                      << std::endl;
            std::cout << "Number of edges: " << getEdgeCount() << std::endl;
            std::cout << "Number of edges with properties:" << edges->edges_with_properties_count() << std::endl;
            std::cout << "Compression Format:" << graph_compr_f_to_string(current_compression) << std::endl;
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
