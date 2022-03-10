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
 * @file agg_sum_all.h
 * @brief Template specializations of the whole-column
 * variants of the aggregation(sum)-operator for uncompressed inputs and
 * outputs using the scalar processing style. Note that these are simple
 * reference implementations not tailored for efficiency.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_All_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_All_H

#include <core/operators/interfaces/agg_sum_all.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace morphstore {
	
	template<>
	struct agg_sum_all_t<vectorlib::scalar<vectorlib::v64<uint64_t>>, uncompr_f, uncompr_f> {
		
		static const column<uncompr_f> *
		apply(const column<uncompr_f> * const in_dataColumm) {
			const size_t in_dataCount = in_dataColumm->get_count_values();
			const uint64_t * const in_data = in_dataColumm->get_data();
			
			// Exact allocation size (for uncompressed data).
			auto out_dataColumn = new column<uncompr_f>(sizeof(uint64_t));
			uint64_t * const out_data = out_dataColumn->get_data();
			
			*out_data = 0;
			for (unsigned i = 0; i < in_dataCount; i ++)
				*out_data += in_data[i];
			
			out_dataColumn->set_meta_data(1, sizeof(uint64_t));
			
			return out_dataColumn;
		}
		
	};
	
	
}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_All_H
