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
 * @file merge_uncompr.h
 * @brief Template specialization of the merge-operator for uncompressed inputs
 * and outputs using the scalar processing style. Note that these are simple
 * reference implementations not tailored for efficiency.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_UNCOMPR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_UNCOMPR_H

#include <core/operators/interfaces/merge.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <vector/scalar/extension_scalar.h>

#include <cstdint>

namespace morphstore {
	
	template<>
	struct merge_sorted_t<vectorlib::scalar<vectorlib::v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f> {
		
		static
		const column<uncompr_f> *
		apply(
		  const column<uncompr_f> * const in_L_posColumn,
		  const column<uncompr_f> * const in_R_posColumn
		) {
			const uint64_t * in_L_pos = in_L_posColumn->get_data();
			const uint64_t * in_R_pos = in_R_posColumn->get_data();
			const uint64_t * const in_L_posEnd = in_L_pos + in_L_posColumn->get_count_values();
			const uint64_t * const in_R_posEnd = in_R_pos + in_R_posColumn->get_count_values();
			
			// If no estimate is provided: Pessimistic allocation size (for
			// uncompressed data), reached only if the two input columns are disjoint.
			auto out_posColumn = new column<uncompr_f>(
			  
			  in_L_posColumn->get_size_used_byte() +
			  in_R_posColumn->get_size_used_byte()
			
			);
			uint64_t * out_pos = out_posColumn->get_data();
			const uint64_t * const initOutPos = out_pos;
			
			while (in_L_pos < in_L_posEnd && in_R_pos < in_R_posEnd) {
				if (*in_L_pos < *in_R_pos) {
					*out_pos = *in_L_pos;
					in_L_pos ++;
				} else if (*in_R_pos < *in_L_pos) {
					*out_pos = *in_R_pos;
					in_R_pos ++;
				} else { // *inPosL == *inPosR
					*out_pos = *in_L_pos;
					in_L_pos ++;
					in_R_pos ++;
				}
				out_pos ++;
			}
			// At this point, at least one of the operands has been fully consumed and
			// the other one might still contain data elements, which must be output.
			while (in_L_pos < in_L_posEnd) {
				*out_pos = *in_L_pos;
				out_pos ++;
				in_L_pos ++;
			}
			while (in_R_pos < in_R_posEnd) {
				*out_pos = *in_R_pos;
				out_pos ++;
				in_R_pos ++;
			}
			
			const size_t outPosCount = out_pos - initOutPos;
			out_posColumn->set_meta_data(outPosCount, outPosCount * sizeof(uint64_t));
			
			return out_posColumn;
		}
	};
}
#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_MERGE_UNCOMPR_H
