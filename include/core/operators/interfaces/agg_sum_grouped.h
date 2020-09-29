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
 * @file agg_sum_grouped.h
 * @brief The template-based interfaces of the group-based aggregation(sum)-operators.
 * @todo Probably, we could generalize the aggregation function using templates somehow.
 */

#ifndef MORPHSTORE_AGG_SUM_GROUPED_H
#define MORPHSTORE_AGG_SUM_GROUPED_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

namespace morphstore {
    
    /**
     * @brief A struct wrapping the actual select-operator. This is necessary to enable
     * partial template specialization.
     */
    template<
      class t_vector_extension,
      class t_out_data_f,
      class t_in_gr_f,
      class t_in_data_f
    >
    struct agg_sum_grouped_t;
    
    
    /**
     * Group-based aggregation(sum)-operator. Aggregates all data elements in the
     * given colum within each of the given groups.
     *
     * Example:
     * - inGrCol:    [  0,   0,  1,   0,   2,   1]
     * - inDataCol:  [100, 150, 50, 500, 200, 100]
     * - inExtCount: 3
     * - outDataCol: [750, 150, 200]
     *
     * @param inGrCol A column whose i-th data element is the group-id of the i-th
     * data element in inDataCol. Must contain as many data elements as inDataCol.
     * @param inDataCol A column containing the data elements to be aggregated
     * based on the group-ids in inGrCol. Must contain the same number of data
     * elements as inGrCol.
     * @param inExtCount The number of groups in the input. Note that this is, in
     * general, not the number of data elements in inGrCol, but the number of data
     * elements in the output extents column from the grouping which produced
     * inGrCol.
     * @return A column of the per-group aggregates. This column contains
     * inExtCount data elements and its i-th data element is the aggregated value
     * of all data elements in inDataCol, whose corresponding group-id in inGrCol
     * equals i.
     */
    template<
      class t_vector_extension,
      class t_out_data_f,
      class t_in_gr_f,
      class t_in_data_f
    >
    const column <t_out_data_f> *
    agg_sum_grouped(
      const column <t_in_gr_f> * const in_groupIdsColumn,
      const column <t_in_data_f> * const in_dataCoumn,
      const size_t in_extendCount
    ) {
        return
          agg_sum_grouped_t<t_vector_extension, t_out_data_f, t_in_gr_f, t_in_data_f>
          ::apply(in_groupIdsColumn, in_dataCoumn, in_extendCount);
    }

} /// namespace morphstore

#endif //MORPHSTORE_AGG_SUM_GROUPED_H
