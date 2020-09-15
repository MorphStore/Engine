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
 * @file page_rank_ldbc_graph_test.cpp
 * @brief Test methods for PageRank on the ldbc graph (only testing csr out of simplicity)
 * @todo
 */
#include <core/operators/graph/page_rank.h>
#include <core/storage/graph/formats/csr.h>
#include <core/storage/graph/importer/ldbc_import.h>
#include <assert.h>

void print_header(std::string storageFormat) {

    std::cout << "\n";
    std::cout << "**********************************************************" << std::endl;
    std::cout << "* MorphStore-Operator-Test: LDBC " << storageFormat << " Page-Rank Test *" << std::endl;
    std::cout << "**********************************************************" << std::endl;
    std::cout << "\n";
}

template <class GRAPH_FORMAT>
void page_rank_ldbc_graph_test (void) {

    static_assert(std::is_base_of<morphstore::Graph, GRAPH_FORMAT>::value, "type parameter of this method must be a graph format");

    std::shared_ptr<morphstore::Graph> graph = std::make_shared<GRAPH_FORMAT>();
    std::string storageFormat = graph->get_storage_format();

    print_header(storageFormat);

    // ldbc importer: path to csv files as parameter: (don't forget the last '/' in adress path)
    std::shared_ptr<morphstore::LDBCImport> ldbcImport = std::make_shared<morphstore::LDBCImport>(LDBC_DIR);

    // generate vertices & edges from LDBC files and insert into graph structure
    ldbcImport->import(*graph);

    // some statistics (DEBUG)
    std::cout << "Some statistics" << std::endl;
    graph->statistics();


    auto result = morphstore::PageRank::compute(graph, 30);

    std::cout << result.describe() << std::endl;

    // TODO: some assertions?
}

int main() {
    page_rank_ldbc_graph_test<morphstore::CSR>();
    return 0;
}