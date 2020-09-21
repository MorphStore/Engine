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
 * @brief Template specializations of the group-related
 * variants of the aggregation(sum)-operator for uncompressed inputs and
 * outputs using the scalar processing style. Note that these are simple
 * reference implementations not tailored for efficiency.
 */

//#include <core/operators/interfaces/agg_sum.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_GROUPED_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_GROUPED_H

namespace morphstore {
	template<>
	struct agg_sum_grouped_t<vectorlib::scalar<vectorlib::v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f> {
		
		static const column <uncompr_f> *
		apply(
		  const column <uncompr_f> * const in_groupIdsColumn,
		  const column <uncompr_f> * const in_dataCoumn,
		  size_t in_extendCount
		) {
			const size_t in_dataCount = in_dataCoumn->get_count_values();
			
			if (in_dataCount != in_groupIdsColumn->get_count_values())
				throw std::runtime_error(
				  "agg_sum: inGrCol and inDataCol must contain the same number "
				  "of data elements"
				);
			
			const uint64_t * const in_groupIds = in_groupIdsColumn->get_data();
			const uint64_t * const in_data = in_dataCoumn->get_data();
			
			const size_t out_dataSize = in_extendCount * sizeof(uint64_t);
			// Exact allocation size (for uncompressed data).
			auto out_dataCol = new column<uncompr_f>(out_dataSize);
			uint64_t * const out_data = out_dataCol->get_data();
			
			for (unsigned i = 0; i < in_extendCount; i ++)
				out_data[i] = 0;
			for (unsigned i = 0; i < in_dataCount; i ++)
				out_data[in_groupIds[i]] += in_data[i];
			
			out_dataCol->set_meta_data(in_extendCount, out_dataSize);
			
			return out_dataCol;
		}
		
	};
}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_GROUPED_H
