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
 * @file graph_compr_format.h
 * @brief helper for specifying compression of graph format specific columns
 * @todo
*/

#ifndef MORPHSTORE_GRAPH_COMPR_FORMAT_H
#define MORPHSTORE_GRAPH_COMPR_FORMAT_H

#include <core/storage/column.h>
#include <core/morphing/morph.h>
#include <core/morphing/format.h>
#include <core/morphing/safe_morph.h>
#include <core/morphing/rle.h>
#include <core/morphing/default_formats.h>
#include <core/morphing/for.h>
#include <core/morphing/k_wise_ns.h>

#include <memory>

namespace morphstore{
    // TODO: allow also other vector extensions (switch from safe_morph to morph)
    // example layout: dynamic_vbp_f<512, 32, 8>
    using ve = vectorlib::scalar<vectorlib::v64<uint64_t>>;

    // TODO use column_base (currently not working as template argument deduction/substitution fails)
    using column_uncompr = column<uncompr_f>;
    using column_delta = column<DEFAULT_DELTA_DYNAMIC_VBP_F(ve)>;
    using column_for = column<DEFAULT_FOR_DYNAMIC_VBP_F(ve)>;
    
    enum class GraphCompressionFormat {DELTA, FOR, UNCOMPRESSED};

    std::string to_string(GraphCompressionFormat format) {
        std::string desc;

        switch (format) {
        case GraphCompressionFormat::DELTA:
            desc = "Delta";
            break;
        case GraphCompressionFormat::UNCOMPRESSED:
            desc = "Uncompressed";
            break;
        case GraphCompressionFormat::FOR:
            desc = "Frame of Reference";
            break;
        }

        return desc;
    }

    // casting the column to the actual column type before morphing (as compiler could not derive it)
    // delete_old_col -> delete input column after morphing (if the result is not the input column)
    const column_base* morph_graph_col(const column_base* column, const GraphCompressionFormat src_f, const GraphCompressionFormat trg_f, bool delete_in_col = false) {
        if (src_f == trg_f) {
            return column;
        }

        const column_base *result;

        switch (src_f) {
        case GraphCompressionFormat::UNCOMPRESSED: {
            const column_uncompr *old_col = dynamic_cast<const column_uncompr *>(column);
            switch (trg_f) {
            case GraphCompressionFormat::DELTA:
                result = morph<ve, DEFAULT_DELTA_DYNAMIC_VBP_F(ve), uncompr_f>(old_col);
                break;
            case GraphCompressionFormat::FOR:
                result = morph<ve, DEFAULT_FOR_DYNAMIC_VBP_F(ve), uncompr_f>(old_col);
                break;
            case GraphCompressionFormat::UNCOMPRESSED:
                result = old_col;
                break;
            }
            return result;
            break;
        }

        // as direct morphing is not yet supported .. go via decompressing first
        case GraphCompressionFormat::DELTA: {
            if (trg_f == GraphCompressionFormat::UNCOMPRESSED) {
                const column_delta *old_col = dynamic_cast<const column_delta *>(column);
                result = morph<ve, uncompr_f, DEFAULT_DELTA_DYNAMIC_VBP_F(ve)>(old_col);
            }
            else {
                auto uncompr_col = morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, delete_in_col);
                result = morph_graph_col(
                    uncompr_col,
                    GraphCompressionFormat::UNCOMPRESSED,
                    trg_f);
                delete uncompr_col;
            }
            break;
        }
        case GraphCompressionFormat::FOR: {
            if (trg_f == GraphCompressionFormat::UNCOMPRESSED) {
                const column_for *old_col = dynamic_cast<const column_for *>(column);
                result = morph<ve, uncompr_f, DEFAULT_FOR_DYNAMIC_VBP_F(ve)>(old_col);
            }
            else {
                auto uncompr_col = morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, delete_in_col);
                result = morph_graph_col(
                    uncompr_col,
                    GraphCompressionFormat::UNCOMPRESSED,
                    trg_f);
                delete uncompr_col;
            }
            break;
        }
        }

        if (result != column && delete_in_col){
            delete column;
        }

        if (result == nullptr) {
            throw std::runtime_error("Did not handle src: " + to_string(src_f) + " trg: " + to_string(trg_f));
        }

        return result; 
    }

    const column_uncompr* decompress_graph_col(const column_base* column, const GraphCompressionFormat src_f, bool delete_in_col = false) {
        return static_cast<const column_uncompr *>(morph_graph_col(column, src_f, GraphCompressionFormat::UNCOMPRESSED, delete_in_col));
    }

    double compression_ratio(const column_base* column, GraphCompressionFormat col_format) {
        // TODO: need to delete decompressed_col? 
        return decompress_graph_col(column, col_format)->get_size_used_byte() / (double) column->get_size_used_byte();
    }
}

#endif //MORPHSTORE_GRAPH_COMPR_FORMAT_H