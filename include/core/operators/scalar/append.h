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
 * @file append.h
 * @brief Implementation of the append operator on replicated columns
 */

#ifndef MORPHSTORE_CORE_OPERATORS_APPEND_H
#define MORPHSTORE_CORE_OPERATORS_APPEND_H

#include <core/operators/interfaces/append.h>

namespace morphstore {

template<>
void append<vectorlib::scalar<vectorlib::v64<uint64_t>>>(replicated_column * const inOriginalCol,

        const size_t value, // could be replaced by the generic datasource provider
        const size_t num
) {
    nonselective_replicated_write_iterator<vectorlib::scalar<vectorlib::v64<uint64_t>>> wi(inOriginalCol);

    pthread_rwlock_wrlock(inOriginalCol->appendLock);

    for (size_t i = 0; i < num; i++)
    {
        wi.write(value); // wi.write(datasource.get())
    }

    wi.done();

    inOriginalCol->logicalCount += num;

    pthread_rwlock_unlock(inOriginalCol->appendLock);

    return;
}


template<>
void append<vectorlib::avx2<vectorlib::v256<uint64_t>>>(replicated_column * const inOriginalCol,

        const size_t value, // could be replaced by the generic datasource provider
        const size_t num
) {
    nonselective_replicated_write_iterator<vectorlib::avx2<vectorlib::v256<uint64_t>>> wi(inOriginalCol);

    pthread_rwlock_wrlock(inOriginalCol->appendLock);

    for (size_t i = 0; i < num; i++)
    {
        wi.write(value); // wi.write(datasource.get())
    }

    wi.done();

    inOriginalCol->logicalCount += num;

    pthread_rwlock_unlock(inOriginalCol->appendLock);

    return;
}

template<>
void append<vectorlib::avx512<vectorlib::v512<uint64_t>>>(replicated_column * const inOriginalCol,

        const size_t value, // could be replaced by the generic datasource provider
        const size_t num
) {
    nonselective_replicated_write_iterator<vectorlib::avx512<vectorlib::v512<uint64_t>>> wi(inOriginalCol);

    pthread_rwlock_wrlock(inOriginalCol->appendLock);

    for (size_t i = 0; i < num; i++)
    {
        wi.write(value); // wi.write(datasource.get())
    }

    wi.done();

    inOriginalCol->logicalCount += num;

    pthread_rwlock_unlock(inOriginalCol->appendLock);

    return;
}

}
#endif // MORPHSTORE_CORE_OPERATORS_APPEND_H