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
 * @file agg_sum_compr_iterator.h
 * @brief A whole-column aggregation operator on compressed data using the 
 * format's read iterator to access the compressed data.
 * @todo Currently, the iterator works only with scalars. When this changes, we
 * should move this file to another directory.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_COMPR_ITERATOR_H
#define MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_COMPR_ITERATOR_H

#include <cstdint>

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/operators/interfaces/agg_sum_all.h>


namespace morphstore {
	/**
	 * TODO: same template specialization as in otfly_derecompr/agg_sum_compr.h -> intended?
	 * shouldn't t_vector_extension be scalar?
	 **/
	template< class t_vector_extension, class t_in_data_f >
	struct agg_sum_all_t<t_vector_extension, uncompr_f, t_in_data_f> {
		
		static
		const column <uncompr_f> *
		apply(const column <t_in_data_f> * const in_dataColumn) {
			const size_t in_dataCount = in_dataColumn->get_count_values();
			
			// Exact allocation size (for uncompressed data).
			auto out_dataColumn = new column<uncompr_f>(sizeof(uint64_t));
			uint64_t * const out_data = out_dataColumn->get_data();
			
			*out_data = 0;
			read_iterator<t_in_data_f> it(in_dataColumn->get_data());
			for (unsigned i = 0; i < in_dataCount; i ++)
				*out_data += it.next();
			
			out_dataColumn->set_meta_data(1, sizeof(uint64_t));
			
			return out_dataColumn;
		}
		
	};
	
}

#endif //MORPHSTORE_CORE_OPERATORS_SCALAR_AGG_SUM_COMPR_ITERATOR_H
