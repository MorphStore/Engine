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
 * @file nvram_allocator.h
 * @brief lightweigth NUMA-aware thread-safe NVRAM memory allocator
 * NVRAM is supposed to be mounted on /mnt/mem0, /mnt/mem1 using DAX feature
 */

#ifndef MORPHSTORE_CORE_STORAGE_NVRAM_ALLOCATOR_H
#define MORPHSTORE_CORE_STORAGE_NVRAM_ALLOCATOR_H

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <unistd.h>
#include <x86intrin.h>
#include <numa.h>
#include <pthread.h>
#include <thread>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace morphstore {

struct nvramChunkData {
    const void* const unalignedPtr;

    size_t fileId = 0;
    size_t fileSize = 0;
    std::string path = "";

    nvramChunkData(const void* unalignedPtr) :
    unalignedPtr(unalignedPtr)
    {
    }
};

std::mutex managmentLock;
std::unordered_map<const void*, const nvramChunkData> nvramChunkDataMap;
std::atomic<size_t> nvramChunkId(0);

void* nvram_alloc(size_t allocSize) {
    void* nvramPointer;
    char nvramChunkFileName[100];
    size_t nvramMappedFileId;
    size_t myNvramChunkId = nvramChunkId++;

    size_t numaSocket;
    numa_available();
    numaSocket = numa_preferred();

    snprintf(nvramChunkFileName, 100, "/mnt/mem%lu/compression_buffer_%lu", numaSocket, myNvramChunkId);
    nvramMappedFileId = open(nvramChunkFileName, O_RDWR | O_CREAT | O_TRUNC, 0);
    posix_fallocate(nvramMappedFileId, 0, allocSize);
    nvramPointer = (void*) mmap(0, allocSize, PROT_READ | PROT_WRITE, MAP_SHARED, nvramMappedFileId, 0);
    memset(nvramPointer, 0, allocSize);
    if (nvramPointer == MAP_FAILED)
    {
        perror("mmap");
    }
    nvramChunkData chunkData(nvramPointer);
    chunkData.fileId = nvramMappedFileId;
    chunkData.fileSize = allocSize;
    chunkData.path = nvramChunkFileName;

    // Ensure thread-safe operation on nvramChunkDataMap
    std::lock_guard<std::mutex> lock(managmentLock);

    nvramChunkDataMap.emplace(nvramPointer, chunkData);

    return nvramPointer;
}

void nvram_free(void* pointer) {

    if (pointer == nullptr) {
        return;
    }

    // Ensure thread-safe operation on nvramChunkDataMap
    std::lock_guard<std::mutex> lock(managmentLock);

    auto it = nvramChunkDataMap.find(pointer);
    if (it == nvramChunkDataMap.end()) {
        return;
    }
    munmap(pointer, it->second.fileSize);
    close(it->second.fileId);
    unlink(it->second.path.c_str());

    nvramChunkDataMap.erase(it);

    return;
}

void nvram_flush(void* m_Data, size_t sizeByte) {
         // Cache processing.
         uint8_t* pos = reinterpret_cast<uint8_t*>(m_Data);
         size_t iters = sizeByte / 64 ;
         for (size_t j = 0; j < iters; j++) {
             _mm_clwb(pos);
             pos += 64;
         }
         _mm_sfence();
}

}
#endif //MORPHSTORE_CORE_STORAGE_NVRAM_ALLOCATOR_H