/**********************************************************************************************
 * Copyright (C) 2020 by MorphStore-Team                                                      *
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
 * @file morph_graph_col.h
 * @brief helper for morphing graph columns (template-free columns). Basically need to cast to template column as it
 * cannot be derieved
 * @todo Remove this helper and make graph formats accept templates (can use normal morph() then)
 */

#ifndef MORPHSTORE_GRAPH_MORPH_GRAPH_COL_H
#define MORPHSTORE_GRAPH_MORPH_GRAPH_COL_H

#include <core/storage/graph/graph_compr_format.h>
#include <core/storage/column.h>
#include <core/morphing/morph.h>

#include <memory>

namespace morphstore {
    using column_uncompr = column<uncompr_f>;
    using column_dyn_vbp = column<default_vbp>;
    using column_delta = column<default_delta>;
    using column_for = column<default_for>;

    // casting the column to the actual column type before morphing (as compiler could not derive it)
    // delete_old_col -> delete input column after morphing (if the result is not the input column)
    const column_base *morph_graph_col(const column_base *column, const GraphCompressionFormat src_f,
                                       const GraphCompressionFormat trg_f, bool delete_in_col = false) {
        if (src_f == trg_f) {
            return column;
        }

        const column_base *result = column;

        switch (src_f) {
        case GraphCompressionFormat::UNCOMPRESSED: {
            const column_uncompr *old_col = dynamic_cast<const column_uncompr *>(column);
            switch (trg_f) {
            case GraphCompressionFormat::DELTA:
                result = morph<ve, default_delta, uncompr_f>(old_col);
                break;
            case GraphCompressionFormat::FOR:
                result = morph<ve, default_for, uncompr_f>(old_col);
                break;
            case GraphCompressionFormat::DYNAMIC_VBP:
                result = morph<ve, default_vbp, uncompr_f>(old_col);
                break;
            case GraphCompressionFormat::UNCOMPRESSED:
                // handled by src_f == trg_f
                break;
            }
            break;
        }
        case GraphCompressionFormat::DELTA: {
            if (trg_f == GraphCompressionFormat::UNCOMPRESSED) {
                const column_delta *old_col = dynamic_cast<const column_delta *>(column);
                result = morph<ve, uncompr_f, default_delta>(old_col);
            } else {
                // as direct morphing is not yet supported .. go via decompressing first
                auto uncompr_col = morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, false);
                result = morph_graph_col(uncompr_col, GraphCompressionFormat::UNCOMPRESSED, trg_f, true);
            }
            break;
        }
        case GraphCompressionFormat::FOR: {
            if (trg_f == GraphCompressionFormat::UNCOMPRESSED) {
                const column_for *old_col = dynamic_cast<const column_for *>(column);
                result = morph<ve, uncompr_f, default_for>(old_col);
            } else {
                // as direct morphing is not yet supported .. go via decompressing first
                auto uncompr_col = morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, false);
                result = morph_graph_col(uncompr_col, GraphCompressionFormat::UNCOMPRESSED, trg_f, true);
            }
            break;
        }
        case GraphCompressionFormat::DYNAMIC_VBP: {
            if (trg_f == GraphCompressionFormat::UNCOMPRESSED) {
                const column_dyn_vbp *old_col = dynamic_cast<const column_dyn_vbp *>(column);
                result = morph<ve, uncompr_f, default_vbp>(old_col);
            } else {
                // as direct morphing is not yet supported .. go via decompressing first
                auto uncompr_col = morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, false);
                // delete_in_col = true as temporary uncompr_col should always be deleted
                result = morph_graph_col(uncompr_col, GraphCompressionFormat::UNCOMPRESSED, trg_f, true);
            }
            break;
        }
        }

        // free input column if possible
        if (result != column && delete_in_col) {
            delete column;
        }

        if (result == nullptr) {
            throw std::runtime_error("Did not handle src: " + graph_compr_f_to_string(src_f) +
                                     " trg: " + graph_compr_f_to_string(trg_f));
        }

        return result;
    }

    const column_uncompr *decompress_graph_col(const column_base *column, const GraphCompressionFormat src_f) {
        return static_cast<const column_uncompr *>(
            morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, false));
    }

    double compression_ratio(const column_base *col, GraphCompressionFormat col_format) {
        auto uncompr_col = decompress_graph_col(col, col_format);
        auto ratio = uncompr_col->get_size_used_byte() / (double)col->get_size_used_byte();

        if (col != uncompr_col) {
            delete uncompr_col;
        }

        return ratio;
    }
} // namespace morphstore

#endif // MORPHSTORE_GRAPH_MORPH_GRAPH_COL_H