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
 * @file basic_operators_interfaces.h
 * @brief The template-based interfaces of the whole-column
 * basic operators.
 * @todo Probably, we could generalize the aggregation function using templates
 * somehow.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_INTERFACES_BASIC_OPERATORS_H
#define MORPHSTORE_CORE_OPERATORS_INTERFACES_BASIC_OPERATORS_H

#include <core/storage/column.h>
#include <core/utils/basic_types.h>

namespace morphstore {

template<
   class VectorExtension,
   class InFormatCol
>
struct agg_sum_t;

/**
 * Whole-column sequential read operator. Aggregates all data elements in the
 * given column.
 *
 * Example:
 * - inDataCol:  [100, 150, 50, 500, 200, 100]
 * - outDataCol: [1100]
 *
 * @param inDataCol A column containing the data elements to be read and aggregated.
 * @return An uncompressed column containing the aggregate value as a single
 * data element.
 */

template<
   class VectorExtension,
   class InFormatCol
>
column<uncompr_f> const *
sequential_read(
   const column<InFormatCol> * const inDataCol
) {
   return agg_sum_t< VectorExtension, InFormatCol >::apply( inDataCol );
}

/**
 * Random read operator. Aggregates all data elements in the
 * given colum following given random indeces.
 * 
 * Example:
 * - inAccessCol:    [ 4,   3,  1,   0,   2,   5]
 * - inDataCol:  [100, 150, 50, 500, 200, 100]
 * - outDataCol: [1100]
 * 
 * @param inAccessCol A column whose i-th data element is the position of the
 * data element in inDataCol. Must contain as many data elements as inDataCol.
 * @param inDataCol A column containing the data elements to be read
 * based on the indeces in inAccessCol. Must contain the same number of data
 * elements as inAccessCol.
 * @return An uncompressed column containing the aggregate value as a single
 * data element.
 */

template<
        class t_vector_extension,
        class t_out_data_f,
        class t_in_access_f,
        class t_in_data_f
>
const column<t_out_data_f> *
random_read(
        const column<t_in_access_f> * const inAccessCol,
        const column<t_in_data_f> * const inDataCol
);

/**
 * Whole-column sequential write operator. Writes all data elements in the
 * given column with sequential integers.
 *
 * Example:
 * - inDataCol before execution: [100, 150, 50, 500, 200, 100]
 * - inDataCol after execution: [0, 1, 2, 3, 4, 5]
 *
 * @param inDataCol A column containing the data elements to be written.
 * @return An uncompressed column same as input containing the modified values.
 */

template<
        class t_vector_extension,
        class t_in_data_f
>
const column<t_in_data_f> *
sequential_write(
        column<t_in_data_f> * const inDataCol
);
//template<
//   class VectorExtension,
//   class InFormatCol
//>
//column<uncompr_f> const *
//sequential_write(
//   column<InFormatCol> * const inDataCol
//);
//
  //return agg_sum_t< VectorExtension, InFormatCol >::apply( inDataCol );
//

/**
 * Random write operator. Writes all data elements in the
 * given column following given random indeces.
 * 
 * Example:
 * - inAccessCol:    [ 4,  3,  1,   0,   2,   5]
 * - inDataCol before execution:  [36, 72, 48, 19, 0, 55]
 * - inDataCol after execution:  [3, 2, 4, 1, 0, 5]
 * 
 * @param inAccessCol A column whose i-th data element is the position of the
 * data element in inDataCol. Must contain as many data elements as inDataCol.
 * @param inDataCol A column containing the data elements to be written
 * based on the indeces in inAccessCol. Must contain the same number of data
 * elements as inAccessCol.
 * @return A modified uncompressed input.
 */

template<
        class t_vector_extension,
        class t_out_data_f,
        class t_in_access_f,
        class t_in_data_f
>
const column<t_out_data_f> *
random_write(
        const column<t_in_access_f> * const inAccessCol,
        column<t_in_data_f> * const inDataCol
);

/**
 * Append operator. Appends given chunk column to original one.
 *
 * Example:
 * - inOriginalCol:    [ 4,   3,  1,   0,   2,   5]
 * - inChunkCol:       [ 10, 15]
 * - InOriginalCol:    [ 4,   3,  1,   0,   2,   5,  10,  15]
 *
 * @param inOriginalCol A column to which chunk is appended.
 * @param inChunkCol A column containing the data elements to be appended at
 * the end of inOriginalCol.
 * @return inOriginalCol exteded by inChunkCol
 */

template<
        class t_vector_extension,
        class t_out_original_f,
        class t_in_original_f,
        class t_in_chunk_f
>
column<t_out_original_f> *
append_chunk(
        column<t_in_original_f> * const inOriginalCol,
        const column<t_in_chunk_f> * const inChunkCol
);

}
#endif //MORPHSTORE_CORE_OPERATORS_INTERFACES_BASIC_OPERATORS_H