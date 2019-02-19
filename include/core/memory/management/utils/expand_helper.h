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
 * @file expand_helper.h
 * @brief Brief description
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_EXPAND_HELPER_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_EXPAND_HELPER_H

#include <core/utils/basic_types.h>
#include <core/utils/math.h>

#include <cassert>

namespace morphstore {

template<size_t MinChunkSize>
constexpr size_t chunk_size(size_t pRequestedSize) {
   static_assert(
      is_power_of_two(MinChunkSize),
      "For performance and convenience granularity (pN) has to be a power of 2.");
   if (MinChunkSize >= pRequestedSize)
      return MinChunkSize;
   size_t remainder = pRequestedSize & (MinChunkSize - 1);
   return (remainder == 0) ? pRequestedSize : pRequestedSize + (MinChunkSize - remainder);
}


size_t chunk_size(size_t MinChunkSize, size_t pRequestedSize) {
   assert(is_power_of_two(MinChunkSize));
   if (MinChunkSize >= pRequestedSize)
      return MinChunkSize;
   size_t remainder = pRequestedSize & (MinChunkSize - 1);
   return (remainder == 0) ? pRequestedSize : pRequestedSize + (MinChunkSize - remainder);
}


class mm_expand_strategy {
   protected:
      size_t mCurrentSize;
   public:
      constexpr mm_expand_strategy(void) : mCurrentSize{0} {}

      constexpr mm_expand_strategy(size_t pCurrentSize) : mCurrentSize{pCurrentSize} {}
};

template<size_t MinimumExpandSize>
class mm_expand_strategy_chunk_based : public mm_expand_strategy {
   static_assert(is_power_of_two(MinimumExpandSize),
                 "For performance and convenience granularity (pN) has to be a power of 2.");
   public:
      constexpr mm_expand_strategy_chunk_based() : mm_expand_strategy(MinimumExpandSize) {}

      inline size_t current_size(void) const {
         return mCurrentSize;
      }

      inline size_t next_size(size_t pExpandSize) {
         mCurrentSize = chunk_size<MinimumExpandSize>(pExpandSize);
         return mCurrentSize;
      }
};

template<size_t MinimumExpandSize>
class mm_expand_strategy_chunk_based_quadratic : public mm_expand_strategy {
   static_assert(is_power_of_two(MinimumExpandSize),
                 "For performance and convenience granularity (pN) has to be a power of 2.");
   public:
      constexpr mm_expand_strategy_chunk_based_quadratic() : mm_expand_strategy(MinimumExpandSize) {}

      inline size_t current_size(void) const {
         return mCurrentSize;
      }

      inline size_t next_size(size_t pExpandSize) {
         mCurrentSize = chunk_size(mCurrentSize << 1, pExpandSize);
         return mCurrentSize;
      }
   };

}
#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_UTILS_EXPAND_HELPER_H

