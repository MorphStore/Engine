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
 * @file ldbc_graph_adjacency.cpp
 * @brief Test for generating social network graph in adj. list format + BFS measurements
 * @todo
 */
#include <core/storage/graph/formats/csr.h>
#include "ldbc_graph_test.h"

int main( void ){
    ldbcGraphFormatTest<morphstore::CSR>();

    // Execute BFS measurements:
    //std::unique_ptr<morphstore::BFS> bfs = std::make_unique<morphstore::BFS>(g1);
    //bfs->do_measurements(10000, "/home/florentin/Morphstore/Output/adj_bfs_SF1.csv");

    return 0;
}
