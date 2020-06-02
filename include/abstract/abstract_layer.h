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
 * @file abstract_layer.h
 * @brief Abstract layer library for managing replicated data
 */

#ifndef MORPHSTORE_ABSTRACT_ABSTRACT_LAYER_H
#define MORPHSTORE_ABSTRACT_ABSTRACT_LAYER_H

#include <vector>
#include "../core/storage/column.h"
#include "../core/storage/column_helper.h"
#include "../core/memory/management/utils/alignment_helper.h"
#include "../core/morphing/format.h"
#include "../core/utils/basic_types.h"
#include "../core/utils/helper_types.h"
#include "../core/storage/replicated_column.h"
#include "../core/storage/nvram_allocator.h"

namespace morphstore {

// Struct to represent the user specified requirements
struct user_params
{
    // Default repica count including master
    size_t replicaCount = 2;
    // DRAM replica placement flag
    bool isVolatileAllowed = false;
    // Commpression flag
    bool isCompressedAllowed = true;
    // Access pattern flah
    bool isSequential = true;

    // Number of predefind config for
    size_t config = 0;
};

// TODO: Class representing cost model, profiles and online state analysis
class cost_model
{

// Struct representing memory consumption per period [1s]. Updated using PCM calls
struct current_state_bandwidth
{
   std::atomic<size_t> dram_read;    // DRAM read BW
   std::atomic<size_t> dram_write;   // DRAM write BW
   std::atomic<size_t> pmm_read;     // PMM read BW
   std::atomic<size_t> pmm_write;    // PMM write BW
};

public:
   // Per socket and intersocket bandwidths statistics assuming 2 socket configuration
   current_state_bandwidth csb1, csb2, csb_upi;

   // Perform test run to obtain profiling data
   void collect_profiles()
   {
   }

   // Load existing profiling data
   void load_profiles()
   {
   }

   // Perform actual decision based on the current state, profiles and user hints
   /*
   static size_t get_optimal(replicated_buffer& rb, user_params& up, format& f)
   {
      return 0;
   }
   */
};

// Utility class containing basic API methods
class ALUtils
{
public:

   static replicated_column* allocate(const size_t size, const user_params& up)
   {
         replicated_buffer* compressed = new replicated_buffer();
         replicated_buffer* uncompressed = new replicated_buffer();

         // Set replica count for compressed and uncompressed replicated buffers
         compressed->m_ReplicatedMetaData.m_ReplicaCount =  up.replicaCount;
         uncompressed->m_ReplicatedMetaData.m_ReplicaCount =  up.replicaCount;

         // Afterwards allocate buffers accordingly
         for (size_t i = 0; i < compressed->m_ReplicatedMetaData.m_ReplicaCount; i++)
         {
            if (compressed->m_ReplicatedMetaData.m_Replicas[i].first == nullptr && uncompressed->m_ReplicatedMetaData.m_Replicas[i].first == nullptr)
            {
                // Apply NUMA policies, for this PoC used round robin style
                numa_set_preferred(i);
                // Select predefind config
                switch (up.config)
                {
                  case 0:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 1:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 2:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 3:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    break;
                  }
                  case 4:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    break;
                  }
                  case 5:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    break;
                  }
                  case 6:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                    if (i == 0)
                    {
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    }
                    else
                    {
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    }
                    break;
                  }
                  case 7:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP;

                    if (i == 0)
                    {
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    }
                    else
                    {
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    }
                    break;
                  }
                  case 8:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 9:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 10:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    break;
                  }
                  case 11:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    break;
                  }
                  case 12:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    if (i == 0)
                    {
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    }
                    else
                    {
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    }
                    break;
                  }
                  case 13:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    if (i == 0)
                    {
                      numa_set_preferred(0);
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    }
                    else if (i == 1)
                    {
                      numa_set_preferred(1);
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    }
                    else if (i == 2)
                    {
                      numa_set_preferred(0);
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    }
                    else if (i == 3)
                    {
                      numa_set_preferred(1);
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR;
                      compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                      uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = false;
                    }
                    break;
                  }
                  case 14:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 15:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 16:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 17:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = DYNAMICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  case 18:
                  {
                    i == 0 ? compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = UNCOMPR : compressed->m_ReplicatedMetaData.m_Replicas[i].second.format = STATICBP;
                    compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM = true;
                    break;
                  }
                  default:
                  {}
                }
                if (compressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM)
                {
                   numa_set_preferred(i);
                   compressed->m_ReplicatedMetaData.m_Replicas[i].first = nvram_alloc(get_size_with_alignment_padding(size));
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_allocated = get_size_with_alignment_padding(size);
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data = 0;
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second.socket = i;
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].first = nvram_alloc(UNCOMPR_SIZE);
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_allocated = UNCOMPR_SIZE;
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data = 0;
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.socket = i;
                }
                else
                {
                   numa_set_preferred(i);
                   compressed->m_ReplicatedMetaData.m_Replicas[i].first = numa_alloc_onnode( get_size_with_alignment_padding(size), i);
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_allocated = get_size_with_alignment_padding(size);
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data = 0;
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second.socket = i;
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].first = numa_alloc_onnode( UNCOMPR_SIZE, i);
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_allocated = UNCOMPR_SIZE;
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data = 0;
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.socket = i;
                }
            }
         }

         return new replicated_column(compressed, uncompressed);
    }

    static replicated_column* take_snapshot(replicated_column* producer)
    {
         replicated_buffer* compressed = new replicated_buffer();
         replicated_buffer* uncompressed = new replicated_buffer();

         // Set replica count for compressed and uncompressed replicated buffers
         compressed->m_ReplicatedMetaData.m_ReplicaCount =  producer->compressed->m_ReplicatedMetaData.m_ReplicaCount;
         uncompressed->m_ReplicatedMetaData.m_ReplicaCount =  producer->uncompressed->m_ReplicatedMetaData.m_ReplicaCount;

         // Lock producer as reader
         pthread_rwlock_rdlock(producer->columnLock);

         // Afterwards configure or allocate buffers accordingly
         for (size_t i = 0; i < compressed->m_ReplicatedMetaData.m_ReplicaCount; i++)
         {
                   compressed->m_ReplicatedMetaData.m_Replicas[i].first = producer->compressed->m_ReplicatedMetaData.m_Replicas[i].first;
                   compressed->m_ReplicatedMetaData.m_Replicas[i].second = producer->compressed->m_ReplicatedMetaData.m_Replicas[i].second;

                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second = producer->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second;

                   // Apply NUMA policies, for this PoC used round robin style
                   numa_set_preferred(uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.socket);
                   if (uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM)
                   {
                       uncompressed->m_ReplicatedMetaData.m_Replicas[i].first = nvram_alloc(UNCOMPR_SIZE);
                   }
                   else
                   {
                       uncompressed->m_ReplicatedMetaData.m_Replicas[i].first = malloc(UNCOMPR_SIZE);
                   }
                   uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_allocated = UNCOMPR_SIZE;
                   memcpy(uncompressed->m_ReplicatedMetaData.m_Replicas[i].first, producer->uncompressed->m_ReplicatedMetaData.m_Replicas[i].first,
                     producer->uncompressed->m_ReplicatedMetaData.m_Replicas[i].second.size_data);
                   // No need to ensure cache flushing for snapshot`s NVRAM uncompressed rest part (it is not a primary data)
         }
         size_t logicalCount = producer->get_logical_count();
         pthread_rwlock_t* lock = producer->columnLock;
         pthread_rwlock_unlock(producer->columnLock);
         return new replicated_column(compressed, uncompressed, true, logicalCount, lock);
      }

};

}      // End of namespace
#endif //MORPHSTORE_ABSTRACT_ABSTRACT_LAYER_H