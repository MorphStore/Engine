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


#ifndef MORPHSTORE_EXTENSION_VIRTUAL_VECTOR_H
#define MORPHSTORE_EXTENSION_VIRTUAL_VECTOR_H

namespace vectorlib {
	
	template<class VirtualVectorView, class VectorExtension>
	struct vv {
		/// check for equality of base type of the virtual vector and the used vector extension
		static_assert(
			std::is_same<
				typename VirtualVectorView::base_t,
				typename VectorExtension::vector_helper_t::base_t
			>::value,
			"Base type of virtual vector and used vector extension mismatch!"
		);
		
		static_assert(
		  VirtualVectorView::size_bit::value >= VectorExtension::vector_helper_t::size_bit::value,
		  "Virtual vector size must be greater or equal to physical vector size!"
        );
		
		static_assert(
		  VirtualVectorView::size_bit::value % VectorExtension::vector_helper_t::size_bit::value == 0,
		  "Virtual vector size must be multiple of physical vector size!"
		);
		
		/// store physically used processing style
		using pps = VectorExtension;
		using vector_t        = typename VirtualVectorView::base_t;
		static_assert(
			std::is_arithmetic<vector_t>::value,
			"Base type of vector register has to be arithmetic."
		);
		
		using vector_helper_t = VirtualVectorView;
		using base_t          = typename vector_helper_t::base_t;
		using size            = std::integral_constant<size_t, sizeof(vector_t)>;
		using mask_t          = uint16_t;
	};
	
}

#endif //MORPHSTORE_EXTENSION_VIRTUAL_VECTOR_H
