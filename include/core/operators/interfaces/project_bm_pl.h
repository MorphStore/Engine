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
 * @file project_bm_pl.h
 * @brief The template-based interface of the project-operator. This interface also handles different, i.e. mixed,
 *        intermediate data representations (position-list + bitmap).
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_BM_PL_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_BM_PL_H

#include <core/storage/column.h>
#include <core/morphing/intermediates/representation.h>

#include <type_traits>

namespace morphstore {

    /**
     * @brief A struct wrapping the actual project-operator.
     *
     * This is necessary to enable partial template specialization, which is
     * required, since some compressed formats have their own template parameters.
     */
    template<
            class t_vector_extension,
            class t_out_data_f,
            class t_in_data_f,
            class t_in_IR_f,
            // enable if actual- & expected-IR-template-parameters are IR-types
            typename std::enable_if_t< is_intermediate_representation_t<t_in_IR_f>::value,int> = 0
    >
    struct project_t {
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
         */
        static
        const column<t_out_data_f> *
        apply(
                const column<t_in_data_f> * const inDataCol,
                const column<t_in_IR_f> * const inPosCol
        ) = delete;
    };

    /**
     * @brief A convenience function wrapping the project-operator.
     *
     * Extracts the data elements at the given positions from the given data
     * column. See documentation of `project_t::apply` for details.
     *
     * @param inDataCol The column to extract from.
     * @param inPosCol_actual Actual intermediate data column (e.g. position-list, bitmap)
     * @return A column containing the extracted data elements
     */
    template<
            class t_vector_extension,
            class t_out_data_f,
            class t_in_data_f,
            class t_in_IR_f,
            // enable if actual- & expected-IR-template-parameters are IR-types
            typename std::enable_if_t<is_intermediate_representation_t<t_in_IR_f>::value,int> = 0
    >
    const column<t_out_data_f> *
    project(
            const column<t_in_data_f> * const inDataCol,
            const column<t_in_IR_f> * const inPosCol
    ) {
        return project_t<t_vector_extension, t_out_data_f, t_in_data_f, t_in_IR_f>::apply(
                inDataCol,
                inPosCol
        );
    }

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_PROJECT_BM_PL_H