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
 * @file project.h
 * @brief The template-based interface of the project-operator.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_H

#include <core/storage/column.h>

namespace morphstore {
    
    /**
     * @brief A struct wrapping the actual select-operator. This is necessary to enable
     * partial template specialization.
    **/
    template<
      class t_vector_extension,
      class t_out_data_f,
      class t_in_data_f,
      class t_in_pos_f
    >
    struct project_t;
    
    /**
     * @brief Project-operator. Extracts the data elements at the given
     * positions from the given data column.
     *
     * Example:
     * - `inDataCol`:  `[11, 44, 22, 33, 11]`
     * - `inPosCol`:   `[     1,      3,  4]`
     * - `outDataCol`: `[    44,     33, 11]`
     *
     * This function is deleted by default, to guarantee that using this struct
     * with a format combination it is not specialized for causes a compiler
     * error, not a linker error.
     *
     * @param inDataCol The column to extract from.
     * @param inPosCol The column containing the positions to extract.
     * @return A column containing the extracted data elements, i.e., as many
     * as in `inPosCol`.
    **/
    template<
      class t_vector_extension,
      class t_out_data_f,
      class t_in_data_f,
      class t_in_pos_f
    >
    const column <t_out_data_f> *
    project(
      const column <t_in_data_f> * const inDataCol,
      const column <t_in_pos_f> * const inPosCol
    ) {
        return
          project_t<t_vector_extension, t_out_data_f, t_in_data_f, t_in_pos_f>
          ::apply(inDataCol, inPosCol);
    }
    
}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_H
