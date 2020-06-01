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
 * @file column_with_blockoffsets.h
 * @brief Wrapper around column + its block-offsets
 */

#ifndef MORPHSTORE_CORE_STORAGE_COLUMN_WITH_BLOCKOFFSETS_H
#define MORPHSTORE_CORE_STORAGE_COLUMN_WITH_BLOCKOFFSETS_H

#include <core/storage/column.h>

namespace morphstore {

// used to allow only partial decompression of column blocks (for random access)
// blockoffsets should only be saved, if blocksize > 1
template <class F> struct column_with_blockoffsets {
    const column<F> *col;
    // TODO: use std::optional
    std::vector<const uint8_t *> *block_offsets;

    column_with_blockoffsets(const column<F> *c) : column_with_blockoffsets(c, new std::vector<const uint8_t *>()) {}

    column_with_blockoffsets(const column<F> *c, std::vector<const uint8_t *> *offsets) {
        col = c;
        block_offsets = offsets;
    }

    ~column_with_blockoffsets() {
        // ? deleting the column might be not always wanted
        delete col;
        delete block_offsets;
    }
};
}
#endif //MORPHSTORE_CORE_STORAGE_COLUMN_WITH_BLOCKOFFSETS_H
