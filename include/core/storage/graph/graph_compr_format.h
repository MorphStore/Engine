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
 * @todo remove need for extra graph-compression format
 */

#ifndef MORPHSTORE_GRAPH_COMPR_FORMAT_H
#define MORPHSTORE_GRAPH_COMPR_FORMAT_H

#include <core/morphing/default_formats.h>
#include <core/morphing/for.h>
#include <core/morphing/format.h>
#include <core/morphing/k_wise_ns.h>
#include <core/morphing/rle.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/delta.h>
#include <core/storage/column.h>

#include <memory>

namespace morphstore {
    // TODO: allow also other vector extensions (regard build flag)
    using ve = vectorlib::scalar<vectorlib::v64<uint64_t>>;

    using default_vbp = DEFAULT_DYNAMIC_VBP_F(ve);
    using default_delta = DEFAULT_DELTA_DYNAMIC_VBP_F(ve);
    using default_for = DEFAULT_FOR_DYNAMIC_VBP_F(ve);

    enum class GraphCompressionFormat { DELTA, FOR, UNCOMPRESSED, DYNAMIC_VBP };

    std::string graph_compr_f_to_string(GraphCompressionFormat format) {
        std::string desc;

        switch (format) {
        case GraphCompressionFormat::DELTA:
            desc = "Delta (Default)";
            break;
        case GraphCompressionFormat::UNCOMPRESSED:
            desc = "Uncompressed";
            break;
        case GraphCompressionFormat::FOR:
            desc = "Frame of Reference (Default)";
            break;
        case GraphCompressionFormat::DYNAMIC_VBP:
            desc = "Dynamic vertical bitpacking (Default)";
            break;
        }

        return desc;
    }
    
    // gets m_BlockSize using the corresponding format  
    // as GraphCompressionFormat is just a simple enum
    size_t inline graph_compr_f_block_size(GraphCompressionFormat format) {
        size_t block_size = 1;

        switch (format) {
        case GraphCompressionFormat::DELTA:
            block_size = default_delta::m_BlockSize;
            break;
        case GraphCompressionFormat::UNCOMPRESSED:
            block_size = uncompr_f::m_BlockSize;
            break;
        case GraphCompressionFormat::FOR:
            block_size = default_for::m_BlockSize;
            break;
        case GraphCompressionFormat::DYNAMIC_VBP:
            block_size = default_vbp::m_BlockSize;
            break;
        }

        return block_size;
    }
} // namespace morphstore

#endif // MORPHSTORE_GRAPH_COMPR_FORMAT_H