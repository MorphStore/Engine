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
 * @file example_query.cpp
 * @brief A little example query making use of almost all query-operators we
 * have so far.
 * @todo TODOS?
 */

#include "core/memory/mm_glob.h"
#include "core/morphing/format.h"

/// operators
#include "core/operators/reference/agg_sum_all.h"
#include "core/operators/reference/agg_sum_grouped.h"
#include "core/operators/reference/agg_sum_compr_iterator.h"
#include "core/operators/uncompr/agg_sum_all.h"
#include "core/operators/reference/group_first.h"
#include "core/operators/reference/group_next.h"
#include "core/operators/reference/join_uncompr.h"
#include "core/operators/reference/project.h"
#include "core/operators/reference/select.h"
#include "core/operators/reference/merge.h"
#include "core/operators/otfly_derecompr/merge.h"

/// storage
#include "core/storage/column.h"
#include "core/storage/column_gen.h"

/// misc
#include "core/utils/basic_types.h"
#include "core/utils/printing.h"
#include "vector/scalar/extension_scalar.h"

/// libs
#include <functional>
#include <iostream>
#include <random>
#include <tuple>

using namespace morphstore;
using namespace vectorlib;

/// ****************************************************************************
/// * Example query
/// ****************************************************************************
///
/// SELECT order.suppKey, order.custKey, SUM(order.qty)
/// FROM order INNER JOIN part ON order.partKey = part.partKey
/// WHERE part.weight < 2000
/// GROUP BY order.suppKey, order.custKey
/// HAVING SUM(order.qty) > 200000
///
/// ****************************************************************************
/// * Basic schema information
/// ****************************************************************************

struct order_t {
    const column<uncompr_f> * suppKey;
    const column<uncompr_f> * custKey;
    const column<uncompr_f> * partKey;
    const column<uncompr_f> * qty;
} order;

struct part_t {
    const column<uncompr_f> * partKey;
    const column<uncompr_f> * weight;
} part;

int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    const size_t orderCount = 1000 * 1000;
    const size_t suppCount  =    1 * 1000;
    const size_t custCount  =    3 * 1000;
    const size_t partCount  =    5 * 1000;
    
    // Supplier/Customer keys are in a certain range, such that we can
    // recognize them in the output.
    const uint64_t minSuppKey = 1 * 1000 * 1000;
    const uint64_t minCustKey = 2 * 1000 * 1000;
    
    order.suppKey = ColumnGenerator::generate_with_distr(
            orderCount,
            std::uniform_int_distribution<uint64_t>(
                    minSuppKey,
                    minSuppKey + suppCount - 1
            ),
            false
    );
    order.custKey = ColumnGenerator::generate_with_distr(
            orderCount,
            std::uniform_int_distribution<uint64_t>(
                    minCustKey,
                    minCustKey + custCount - 1
            ),
            false
    );
    order.partKey = ColumnGenerator::generate_with_distr(
            orderCount,
            std::uniform_int_distribution<uint64_t>(0, partCount - 1),
            false
    );
    order.qty = ColumnGenerator::generate_with_distr(
            orderCount,
            std::uniform_int_distribution<uint64_t>(10, 100000),
            false
    );
    
    part.partKey = ColumnGenerator::generate_sorted_unique(partCount);
    part.weight = ColumnGenerator::generate_with_distr(
            partCount,
            std::uniform_int_distribution<uint64_t>(1000, 5000),
            false
    );
    
    std::cout << "done." << std::endl;
    
    // ************************************************************************
    // * Query execution
    // ************************************************************************
    
    using ve = scalar<v64<uint64_t> >;
    
    std::cout << "Query execution started... ";
    std::cout.flush();
    
    // It is really hard to give the intermediates sensible names...
    // This plan was created by hand. Probably there is a better one.
    
    // Positions in "part" fulfilling "part.weight < 2000"
    auto iPPos = morphstore::select<
            ve,
            vectorlib::less,
            uncompr_f,
            uncompr_f
    >(part.weight, 2000);
    
    // Data elements of "part.partKey" fulfilling "part.weight < 2000"
    auto iPPartKeyProj = project<ve, uncompr_f>(part.partKey, iPPos);
    
    // Positions in "order" having a join partner in the filtered "part"
    // relation
    const column<uncompr_f> * iOJoinedPos;
    // (unused)
    const column<uncompr_f> * iPJoinedPos;
    std::tie(iOJoinedPos, iPJoinedPos) = nested_loop_join<
            ve,
            uncompr_f,
            uncompr_f
    >(order.partKey, iPPartKeyProj, orderCount);
    
    // "order.suppKey", "order.custKey", and "order.qty" for the positions in
    // "order" which have a join partner in "part"
    auto iOSuppKeyProj = project<ve, uncompr_f>(order.suppKey, iOJoinedPos);
    auto iOCustKeyProj = project<ve, uncompr_f>(order.custKey, iOJoinedPos);
    auto iOQtyProj     = project<ve, uncompr_f>(order.qty    , iOJoinedPos);
    
    // First step of the grouping: group by "order.suppKey".
    // Group-ids of *unary* grouping on filtered "order.suppKey"
    const column<uncompr_f> * iOSuppKeyGr;
    // Positions of group-representatives of *unary* grouping on filtered
    // "order.suppKey"
    const column<uncompr_f> * iOSuppKeyExt;
    std::tie(iOSuppKeyGr, iOSuppKeyExt) = group_first<ve, uncompr_f, uncompr_f, uncompr_f>(
            iOSuppKeyProj
    );
    
    // Second step of the grouping: additionally group by "order.custKey".
    // Group-ids of *binary* grouping on filtered "order.custKey"
    const column<uncompr_f> * iOCustKeyGr;
    // Positions of group-representatives of *binary* grouping on filtered
    // "order.custKey"
    const column<uncompr_f> * iOCustKeyExt;
    std::tie(iOCustKeyGr, iOCustKeyExt) = group_next<ve, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(
            iOSuppKeyGr, iOCustKeyProj
    );
    
    // Per-group aggregates of filtered "order.qty"
    auto iOQtyGrSum = agg_sum_grouped<ve, uncompr_f>(
            iOCustKeyGr,
            iOQtyProj,
            iOCustKeyExt->get_count_values()
    );
    
    // Group-ids of *binary* grouping fulfilling "SUM(order.qty) > 200000"
    auto iOPosHaving = morphstore::select<
            ve,
            vectorlib::greater,
            uncompr_f,
            uncompr_f
    >(iOQtyGrSum, 200000);
    
    // Per-group aggregates of filtered "order.qty" fulfilling
    // "SUM(order.qty) > 200000"
    auto iOQtyGrSumHaving = project<ve, uncompr_f>(iOQtyGrSum, iOPosHaving);
    
    // Looking up the representatives of "order.custKey" (the *second* grouping
    // attribute).
    // Positions of group-representatives fulfilling "SUM(order.qty) > 200000"
    auto iOCustKeyHavingRepr0 = project<ve, uncompr_f>(
            iOCustKeyExt,
            iOPosHaving
    );
    // Group-representatives fulfilling "SUM(order.qty) > 200000"
    auto iOCustKeyHavingRepr = project<ve, uncompr_f>(
            iOCustKeyProj,
            iOCustKeyHavingRepr0
    );
    
    // Looking up the representatives of "order.suppKey" (the *first* grouping
    // attribute).
    // Group-ids of the *unary* grouping belonging to the group-ids of the
    // *binary* grouping fulfilling "SUM(order.qty) > 200000"
    auto iOSuppKeyHavingRepr0 = project<ve, uncompr_f>(
            iOSuppKeyGr,
            iOCustKeyHavingRepr0
    );
    // Positions of group-representatives fulfilling "SUM(order.qty) > 200000"
    auto iOSuppKeyHavingRepr1 = project<ve, uncompr_f>(
            iOSuppKeyExt,
            iOSuppKeyHavingRepr0
    );
    // Group-representatives fulfilling "SUM(order.qty) > 200000"
    auto iOSuppKeyHavingRepr  = project<ve, uncompr_f>(
            iOSuppKeyProj,
            iOSuppKeyHavingRepr1
    );
    
    std::cout << "done." << std::endl << std::endl;
    
    // ************************************************************************
    // * Result output
    // ************************************************************************
    
    print_columns(
            print_buffer_base::decimal,
            iOSuppKeyHavingRepr,
            iOCustKeyHavingRepr,
            iOQtyGrSumHaving,
            "order.suppKey",
            "order.custKey",
            "SUM(order.qty)"
    );
    
    return 0;
}
