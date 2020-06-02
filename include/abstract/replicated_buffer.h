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
 * @file replicated_buffer.h
 * @brief General container class for managing buffers representing replicated column
 */

#include <vector>
#include <utility>
#include <core/storage/column.h>
#include <core/storage/column_helper.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>
#include <abstract/abstract_layer.h>

#ifndef MORPHSTORE_ABSTRACT_REPLICATED_BUFFER_H
#define MORPHSTORE_ABSTRACT_REPLICATED_BUFFER_H

#define UNCOMPR_SIZE 4096     // Predefined size of uncompressed rest buffers - must be cache line size alligned
#define MAX_REPLICA_COUNT 8   // Predefined max count of replicas per replicated buffer

namespace morphstore {

// Forward declaration
struct user_params;

// Abstract library data formats as enumaration, while the numbers are to be used on other levels
enum buffer_format
{
   UNCOMPR = 0, STATICBP = 1, DYNAMICBP = 2
};

struct replicated_buffer
{

// Meta data struct describing an elementary buffer
struct buffer_meta_data
{
   bool isMaster = false;
   bool isNVRAM = true;
   buffer_format format = STATICBP;
   size_t socket = 0;
   size_t size_allocated = 0;
   size_t size_data = 0;
};

// Meta data struct describing a replicated buffer as a whole
struct replicated_buffer_meta_data
{
   size_t m_ReplicaCount;
   std::vector< std::pair<void*, buffer_meta_data> > m_Replicas;

   replicated_buffer_meta_data() : m_Replicas(MAX_REPLICA_COUNT)
   {
      m_ReplicaCount = MAX_REPLICA_COUNT;
      buffer_meta_data bmd;
      for (size_t i = 0; i < m_ReplicaCount; i++)
      {
          m_Replicas.push_back(std::make_pair(nullptr,bmd));
      }
   }

   replicated_buffer_meta_data(replicated_buffer_meta_data &&) = delete;
   replicated_buffer_meta_data& operator=(replicated_buffer_meta_data const&) = delete;
   replicated_buffer_meta_data& operator=(replicated_buffer_meta_data&& that) = delete;
   ~replicated_buffer_meta_data()
   {
   }
};

      replicated_buffer_meta_data m_ReplicatedMetaData;

      // Create an empty replicated buffer
      replicated_buffer(): m_ReplicatedMetaData()
      {
      }

      // The following methods are not provided
      replicated_buffer(replicated_buffer &&) = delete;
      replicated_buffer& operator= (replicated_buffer const&) = delete;
      replicated_buffer& operator= (replicated_buffer &&) = delete;

      // Return the raw pointer to the respective replica
      voidptr_t get_buffer(size_t replicaNumber)
      {
         return m_ReplicatedMetaData.m_Replicas[replicaNumber].first;
      }

      // TODO: Return the number of replica considered the best for current query by the cost model
      size_t get_optimal(user_params& up)
      {
         size_t num_opt = 0;
         //num_opt = morphstore::cost_model::get_opt(up);
         return num_opt;
      }

      // Destructor. Here the allocated memory (both DRAM and NVRAM) is freed
      virtual ~replicated_buffer()
      {
         // Delete actually allocated memory buffers
         for (size_t i = 0; i < m_ReplicatedMetaData.m_ReplicaCount; i++)
         {
            if (m_ReplicatedMetaData.m_Replicas[i].first != nullptr)
            {
               if (m_ReplicatedMetaData.m_Replicas[i].second.isNVRAM)
                   nvram_free(m_ReplicatedMetaData.m_Replicas[i].first);
               else
               {
                   numa_free(m_ReplicatedMetaData.m_Replicas[i].first, m_ReplicatedMetaData.m_Replicas[i].second.size_allocated);
               }
            }
         }
      }

};

}      // End of morphstore namespace
#endif //MORPHSTORE_ABSTRACT_REPLICATED_BUFFER_H