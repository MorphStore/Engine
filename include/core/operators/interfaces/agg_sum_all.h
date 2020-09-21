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
 * @file agg_sum.h
 * @brief The template-based interfaces of the whole-column aggregation(sum)-operators.
 * @todo Probably, we could generalize the aggregation function using templates somehow.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_AGG_SUM_ALL_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_AGG_SUM_ALL_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

namespace morphstore {
	
	template<
	  class t_vector_extension,
	  class t_out_data_f,
	  class t_in_data_f
	>
	struct agg_sum_all_t;
	
	/**
	 * Whole-column aggregation(sum)-operator. Aggregates all data elements in the
	 * given column.
	 *
	 * Example:
	 * - inDataCol:  [100, 150, 50, 500, 200, 100]
	 * - outDataCol: [1100]
	 *
	 * @param inDataCol A column containing the data elements to be aggregated.
	 * @return An uncompressed column containing the aggregate value as a single
	 * data element. We always use the uncompressed format here, since compressing
	 * a single value would not make much sense.
	***/
	template<
	  class t_vector_extension,
	  class t_out_data_f,
	  class t_in_data_f
	>
	column<t_out_data_f> const *
	agg_sum_all(const column<t_in_data_f> * const in_dataColumn) {
		return agg_sum_all_t<t_vector_extension, t_out_data_f, t_in_data_f>::apply(in_dataColumn);
	}
}


#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_AGG_SUM_ALL_H


