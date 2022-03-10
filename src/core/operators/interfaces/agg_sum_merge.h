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


#ifndef QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_OPERATORS_AGG_SUM_MERGE_H
#define QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_OPERATORS_AGG_SUM_MERGE_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

#include <tuple>

namespace morphstore {
    template<
      class TVectorExtension,
      class TOutGroupsF,
      class TOutSumF,
      class TInGroupsF,
      class TInSumF
    >
    struct agg_sum_merge_t;
    
    
    template<
      class TVectorExtension,
      class TOutGroupsF,
      class TOutSumF,
      class TInGroupsF,
      class TInSumF
    >
    const std::tuple<
      const column <TOutGroupsF> *,
      const column <TOutSumF> *
    >
    agg_sum_merge(
      column <TInGroupsF> const * const * const in_groups,
      column <TInSumF>    const * const * const in_sums,
      const uint64_t partitions
    ) {
        return agg_sum_merge_t<
          TVectorExtension,
          TOutGroupsF,
          TOutSumF,
          TInGroupsF,
          TInSumF
        >::apply(in_groups, in_sums, partitions);
    };
} /// namespace morphstore
#endif //QUEUEBENCHMARK_INCLUDE_MORPHSTORE_INCLUDE_CORE_OPERATORS_AGG_SUM_MERGE_H
