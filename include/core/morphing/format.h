/**********************************************************************************************
 * Copyright (C) 2019 by Patrick Damme                                                        *
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
 * @file format.h
 * @brief Brief description
 * @author Patrick Damme
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MORPHING_FORMAT_H
#define MORPHSTORE_CORE_MORPHING_FORMAT_H


namespace morphstore {

// TODO don't forget to make this abstract (when adding member functions)
// The abstract base of all formats.
struct format {
    //
};

// The uncompressed format.
struct uncompr_f : public format {
    //
};

}
#endif //MORPHSTORE_CORE_MORPHING_FORMAT_H