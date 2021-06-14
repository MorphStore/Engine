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
using namespace std;


int main(){
    using ve = scalar<v64<uint64_t>>;
    auto col0 = ColumnGenerator::generate_with_distr(
              10000,
              std::uniform_int_distribution<uint64_t>(0, 100),
              false,
              8
            );
    auto col1 = ColumnGenerator::generate_with_distr(
              10000,
              std::uniform_int_distribution<uint64_t>(0, 100),
              false,
              7
            );
    
    auto & [gid1, gext1] = group_first<ve, uncompr_f, uncompr_f, uncompr_f>(col0);
    auto & [gid2, gext2] = group_next<ve,  uncompr_f, uncompr_f, uncompr_f, uncompr_f>(gid1, col1);
    
    auto & [gid, gext] = group_next<ve, uncompr_f, uncompr_f, uncompr_f, uncompr_f>(col0, col1);
    
    
    uint64_t * bdata0 = col0->get_data();
    uint64_t * bdata1 = col1->get_data();
    uint64_t * gdata1 = gid2->get_data();
    uint64_t * gdata2 = gid->get_data();
    
    if(gid1->get_count_values() != gid->get_count_values()){
        error("Element count differs!");
    }
    
    bool equal = true;
    for(uint64_t i = 0; i < gid1->get_count_values(); ++i){
        // debug msg:
//        cout << "(" << bdata0[i] << ", " << bdata1[i] << ") ->" << gdata1[i] << " : " << gdata2[i] << endl;
        if(gdata1[i] != gdata2[i])
            equal = false;
    }
    
    if(!equal){
        error("Result mismatch");
    }
    
    return 0;
}
