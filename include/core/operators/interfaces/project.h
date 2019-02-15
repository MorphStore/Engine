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
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_H

#include "../../storage/column.h"
#include "../../utils/processing_style.h"

namespace morphstore {
    
/**
 * Project-operator. Extracts the data elements at the given positions from the
 * given data column.
 * 
 * Example:
 * - inDataCol:  [11, 44, 22, 33, 11]
 * - inPosCol:   [     1,      3,  4]
 * - outDataCol: [    44,     33, 11]
 * 
 * @param inDataCol The column to extract from.
 * @param inPosCol The column containing the positions to extract.
 * @return A column containing the extracted data elements, i.e., as many as in
 * inPosCol.
 */
template<
        processing_style_t t_ps,
        class t_out_data_f,
        class t_in_data_f,
        class t_in_pos_f
>
const column<t_out_data_f> *
project(
        const column<t_in_data_f> * const inDataCol,
        const column<t_in_pos_f> * const inPosCol
);

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_H