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
 * @file replicated_column.h
 * @brief General class for managing replicated columns
 */

#include <vector>
#include <core/storage/column.h>
#include <core/storage/column_helper.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/utils/basic_types.h>
#include <core/utils/helper_types.h>
#include <abstract/abstract_layer.h>
#include <abstract/replicated_buffer.h>

#ifndef MORPHSTORE_CORE_STORAGE_REPLICATED_COLUMN_H
#define MORPHSTORE_CORE_STORAGE_REPLICATED_COLUMN_H

namespace morphstore {

struct replicated_column
{
      size_t logicalCount;
      replicated_buffer* compressed;
      replicated_buffer* uncompressed;
      pthread_rwlock_t* columnLock;
      pthread_rwlock_t* appendLock;
      bool isSnapshot = false;

      // Create replicated column as wrapper for compressed and uncompressed buffers
      replicated_column(replicated_buffer* p_compressed, replicated_buffer* p_uncompressed)
      {
         this->logicalCount =  0; // Assumed initially empty buffers
         this->compressed =  p_compressed;
         this->uncompressed =  p_uncompressed;
         columnLock = new pthread_rwlock_t();
         pthread_rwlock_init(columnLock, NULL);
         appendLock = new pthread_rwlock_t();
         pthread_rwlock_init(appendLock, NULL);
      }

      // Create replicated column as a snaphot
      replicated_column(replicated_buffer* p_compressed, replicated_buffer* p_uncompressed, bool p_IsSnapshot, size_t p_LogicalCount, pthread_rwlock_t* p_ColumnLock)
      {
         this->logicalCount = p_LogicalCount;
         this->compressed =  p_compressed;
         this->uncompressed =  p_uncompressed;
         this->isSnapshot = p_IsSnapshot;
         this->columnLock = p_ColumnLock;
      }

      // Avoid compilation
      replicated_column(replicated_column &&) = delete;
      replicated_column& operator= (replicated_column const&) = delete;
      replicated_column& operator= (replicated_column &&) = delete;

      // Get logical count
      size_t get_logical_count()
      {
         return logicalCount;
      }

      // Get raw pointer to the uncompressed part of the reqested replica
      voidptr_t get_comressed_data(size_t replicaNumber)
      {
         return compressed->m_ReplicatedMetaData.m_Replicas[replicaNumber].first;
      }

      // Get raw pointer to the uncompressed part of the reqested replica
      voidptr_t get_uncomressed_data(size_t replicaNumber)
      {
         return uncompressed->m_ReplicatedMetaData.m_Replicas[replicaNumber].first;
      }

      // Returns targeted column by replica number
      template<class t_ve>
      void* get_column(size_t num, size_t& format)
      {
         size_t sizeData, sizeAllocated, sizeUncomprData;  // Assume compressed buffer is of a fixed known UNCOMPR_SIZE size
         bool isNVRAM;

         format = compressed->m_ReplicatedMetaData.m_Replicas[num].second.format;
         sizeData = compressed->m_ReplicatedMetaData.m_Replicas[num].second.size_data;
         sizeAllocated = compressed->m_ReplicatedMetaData.m_Replicas[num].second.size_allocated;
         sizeUncomprData = uncompressed->m_ReplicatedMetaData.m_Replicas[num].second.size_data;
         isNVRAM = compressed->m_ReplicatedMetaData.m_Replicas[num].second.isNVRAM;

         switch (format)
         {
          case 0: // UNCOMPR
          {
             return (void*)(new column<uncompr_f>(compressed->m_ReplicatedMetaData.m_Replicas[num].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
          }
          case 1: // STATICBP 32
          {
             return (void*)(new column<DEFAULT_STATIC_VBP_F(t_ve, 32)>(compressed->m_ReplicatedMetaData.m_Replicas[num].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
          }
          case 2: // DYNAMICBP
          {
             return (void*)(new column<DEFAULT_DYNAMIC_VBP_F(t_ve)>(compressed->m_ReplicatedMetaData.m_Replicas[num].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
          }
         }

         return (void*)(new column<uncompr_f>(compressed->m_ReplicatedMetaData.m_Replicas[num].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
      }

      // Returns targeted column automatically selected among replicas for the current query
      template<class t_ve>
      void* get_column(size_t& format, user_params* up = nullptr)
      {
         size_t num_optimal, sizeData, sizeAllocated, sizeUncomprData;
         bool isNVRAM;

         //TODO: use cost_model num_optimal = compressed->get_optimal(up);
         uint32_t cpu, node;
         cpu = sched_getcpu();
         node = numa_node_of_cpu(cpu);
         num_optimal = node;
         format = compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].second.format;
         sizeData = compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].second.size_data;
         sizeAllocated = compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].second.size_allocated;
         sizeUncomprData = uncompressed->m_ReplicatedMetaData.m_Replicas[num_optimal].second.size_data;
         isNVRAM = compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].second.isNVRAM;
         numa_set_preferred(num_optimal);
         switch (format)
         {
          case 0: // UNCOMPR
          {
             return (void*)(new column<uncompr_f>(compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
          }
          case 1: // STATICBP 32
          {
             return (void*)(new column<DEFAULT_STATIC_VBP_F(t_ve, 32)>(compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
          }
          case 2: // DYNAMICBP
          {
             return (void*)(new column<DEFAULT_DYNAMIC_VBP_F(t_ve)>(compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
          }
         }

         return (void*)(new column<uncompr_f>(compressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, uncompressed->m_ReplicatedMetaData.m_Replicas[num_optimal].first, logicalCount, sizeData, sizeAllocated, sizeUncomprData, isNVRAM));
      }

      // Destructor. Actual memory chunks are cleaned by the replicated_buffer container
      virtual ~replicated_column( )
      {
         if (!isSnapshot)
         {
             delete compressed;
             pthread_rwlock_destroy(columnLock);
             delete columnLock;
             pthread_rwlock_destroy(appendLock);
             delete appendLock;
         }
         // Delete uncompressed part anyway
         delete uncompressed;
      }

};

}      // End of morphstore namespace
#endif //MORPHSTORE_CORE_STORAGE_REPLICATED_COLUMN_H