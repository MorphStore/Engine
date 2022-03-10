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
 * @file noselfmanaging_helper.h
 * @brief A utility for ensuring that self-managed memory is turned off.
 */

#ifndef MORPHSTORE_CORE_MEMORY_NOSELFMANAGING_HELPER_H
#define MORPHSTORE_CORE_MEMORY_NOSELFMANAGING_HELPER_H

#include <stdexcept>


namespace morphstore {

    // @todo This should not be required anywhere. Remove this feature as soon
    // as MorphStore's memory manager is fit for all our use cases.
    void fail_if_self_managed_memory() {
#ifndef MSV_NO_SELFMANAGED_MEMORY
        throw std::runtime_error(
                "Currently, this executable only works with non-self-managed "
                "memory. Compile MorphStore with build.sh -noSelfManaging ."
        );
#endif
    }
    
}
#endif //MORPHSTORE_CORE_MEMORY_NOSELFMANAGING_HELPER_H
