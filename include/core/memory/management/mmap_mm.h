#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H

#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
#  error "Abstract memory manager ( management/abstract_mm.h ) has to be included before mmap memory manager."
#endif

#include <core/memory/management/abstract_mm.h>

#include <mutex>
#include <sys/mman.h>
#include <string.h>

#include <cassert>

namespace morphstore {

const size_t LINUX_PAGE_SIZE = 1 << 12;
const size_t DB_PAGE_SIZE = 1 << 14;
const size_t DB_PAGE_OFFSET = 14;
const size_t ALLOCATION_SIZE = 1l << 27;
const size_t ALLOCATION_OFFSET = 27;

enum StorageType : uint64_t {
    CONTINUOUS,
    LARGE
};

class ChunkHeader;

class AllocationStatus {
public:
AllocationStatus() : next(nullptr), curr_offset(0), type(StorageType::CONTINUOUS) {}
    ChunkHeader* next;
    uint64_t curr_offset;
    std::mutex sema;
    StorageType type;
};

class InfoHeader {
public:
    char unused[LINUX_PAGE_SIZE - sizeof(AllocationStatus)];
    AllocationStatus status;
};
static_assert(sizeof(InfoHeader) == 4096, "Assuming info header should be page size'd");

class ChunkHeader {
    //place holder
    //size must be aligned to linux page size
    // Bitfield is used as follows: one entry for each page is 2 bit
    // UNUSED  = 00
    // UNMAP   = 01
    // START   = 10
    // CONT    = 11
public:
    static const uint64_t ALLOC_BITS = 2;
    static const uint8_t UNUSED = 0;

    void initialize(StorageType type)
    {
        info.status.next = nullptr;
        info.status.curr_offset = 0;
        info.status.type = type;
    }

    void reset()
    {
        memset(bitmap, UNUSED, LINUX_PAGE_SIZE);
    }

    void* getCurrentOffset()
    {
        size_t chunkLocation = reinterpret_cast<size_t>(this) + sizeof(ChunkHeader);
        return reinterpret_cast<void*>(chunkLocation + reinterpret_cast<size_t>(info.status.curr_offset));
    }

    // TODO: implement atomic variant to enable lock-free allocation
    // Not used yet
    void forwardOffset(size_t size)
    {
        char* address = reinterpret_cast<char*>(getCurrentOffset());
        address += size;
        //char* chunkLocation = reinterpret_cast<char*>(this) + sizeof(ChunkHeader);
        info.status.curr_offset = reinterpret_cast<uint64_t>(address);
    }

    void setAllocated(void* address, size_t size)
    {
        char* location_this = reinterpret_cast<char*>(this);
        uint64_t location_in_chunk = reinterpret_cast<uint64_t>(address) - reinterpret_cast<uint64_t>(location_this + sizeof(ChunkHeader));
        location_in_chunk = location_in_chunk / DB_PAGE_SIZE;

        long needed_blocks = size / static_cast<long>(DB_PAGE_SIZE);
        if (size % DB_PAGE_SIZE != 0)
            ++needed_blocks;

        // Calculate precise location in bitmap
        uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
        //uint64_t byte_pos_in_bitmap = bit_offset >> 3;
        uint64_t word_aligned_pos_in_bitmap = bit_offset >> 5;
        uint64_t bit_offset_in_word = bit_offset & 0x1fl;

        long allocate_in_word = 32 - bit_offset_in_word / 2;
        
        uint64_t* word_in_bitmap = reinterpret_cast<uint64_t*>(bitmap + word_aligned_pos_in_bitmap);
        uint64_t state = getAllocBits64(bit_offset_in_word, needed_blocks);

        *word_in_bitmap |= state;
        needed_blocks -= allocate_in_word;
        // Move forward in bitmap by one word
        word_in_bitmap = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(word_in_bitmap) + sizeof(uint64_t));
        while (needed_blocks > 0) {
             state = getAllocBits64(0, needed_blocks, false);
             *word_in_bitmap = state;
             needed_blocks -= sizeof(uint64_t);
             word_in_bitmap = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(word_in_bitmap) + sizeof(uint64_t));
        }
    }

    void setDeallocated(void* address)
    {
        // TODO: just do it
        uint64_t location_this = reinterpret_cast<uint64_t>(this);
        uint64_t location_in_chunk = reinterpret_cast<uint64_t>(address) - (location_this + sizeof(ChunkHeader));
        location_in_chunk = location_in_chunk / DB_PAGE_SIZE;

        // Calculate precise location in bitmap
        uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
        uint64_t word_aligned_pos_in_bitmap = bit_offset >> 5;
        uint64_t bit_offset_in_word = bit_offset & 0x1fl;

        // Find out length of allocated word
        //TODO: rest
        // Mask which starts with 1s at the first allocation bits
        //uint64_t mask_for_allocation = (1 << (bit_offset_in_word)) - 1;
        uint64_t* word_in_bitmap = reinterpret_cast<uint64_t*>(bitmap + word_aligned_pos_in_bitmap);

        // Get allocation length
        uint64_t bit_position_for_cont = bit_offset_in_word + ALLOC_BITS;
        uint64_t mask_for_cont = ((1 << bit_position_for_cont) - 1) & 0x5555555555555555; // constant for identifying continuation
        
        uint64_t copy = *word_in_bitmap;
        copy = copy & mask_for_cont;
        bool endFound = false;

        for (uint64_t i = bit_offset_in_word + ALLOC_BITS; !endFound; i += ALLOC_BITS) {
            uint64_t* word = reinterpret_cast<uint64_t*>(i >> 5);
            if ( ((*word >> bit_offset_in_word) & 0b11) == 0b11 )
                *word = *word & ~(0l + (0b11 << bit_offset_in_word));
            else
                endFound = true;
        }

    }

    // used to decide whether we can reallocate in front or behind a memory location
    bool isAllocatable(void* /*start*/, size_t /*size*/)
    {
        //TODO: implement
        return true;
    }

    void* findNextAllocatableSlot(size_t size)
    {
        // TODO: replace with strategy pattern
        uint64_t* loc = reinterpret_cast<uint64_t*>(bitmap);
        uint64_t slots_needed = (size >> DB_PAGE_OFFSET) + ( ((size % DB_PAGE_SIZE) == 0) ? 0 : 1);

        uint64_t continuousPageCounter = 0;
        //bool runningContinuous = false;
        uint64_t bit_start = 0;

        while (reinterpret_cast<char*>(loc) < bitmap + LINUX_PAGE_SIZE) {
            if (*loc < ~(0x8l << 60)) {
                uint64_t bit_offset = 0;

                while (bit_offset < 64) {
                    uint8_t bits = *loc >> (64 - bit_offset) & 0b11l;
                    if (bits == 0) {
                        //runningContinuous = true;
                        //uint64_t bit_start = reinterpret_cast<char*>(loc) - bitmap + bit_offset;
                        ++continuousPageCounter;
                        if (continuousPageCounter >= slots_needed) {
                            void* addr = this + sizeof(ChunkHeader) + DB_PAGE_SIZE * (bit_start / 2);
                            return addr; 
                        }
                    }
                }
                loc = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(loc) + sizeof(uint64_t));
            }
            else {
                //runningContinuous = false;
                bit_start = 0;
                continuousPageCounter = 0;
            }
        }
    }

    char bitmap[LINUX_PAGE_SIZE];
    InfoHeader info;

private:
    inline uint64_t getAllocBits64(uint8_t startOffset, uint32_t count_blocks, bool isStart = true)
    {
        uint8_t i = startOffset;
        uint64_t state = 0;

        // Start is marked directly at first two MSB
        if (startOffset == 0 && isStart)
            return (~(0l) - (1l << 63));

        // TODO: optimize, dunno if compiler optimizes
        while (i < 64/ALLOC_BITS && count_blocks > 0) {
            state |= (i == startOffset && isStart ? 0b10 : 0b11) << (64 - i*ALLOC_BITS);
            --count_blocks;
            ++i; 
        }

        return state;
    }
};
static_assert(sizeof(ChunkHeader) == 8192, "Code expects ChunkHeader to be two pages long, please refit solution in case that should not hold");

const size_t HEAD_STRUCT = sizeof(ChunkHeader);
const size_t IDEAL_OFFSET = ALLOCATION_SIZE - HEAD_STRUCT;

class mmap_memory_manager: public abstract_memory_manager {
public:
    static mmap_memory_manager* m_Instance;
    static mmap_memory_manager& getInstance()
    {
        if (m_Instance == nullptr)
            m_Instance = ::new mmap_memory_manager();
        return *m_Instance;
    }
    ~mmap_memory_manager() {}

    /**
     * @brief use for continuous memory allocation
     */
    void* allocateContinuous()
    {
        // need at least 2*alloc_size-1 for guaranteed alignment
        const size_t mmap_alloc_size = 2 * ALLOCATION_SIZE + HEAD_STRUCT;
        char* given_ptr = reinterpret_cast<char*>(mmap(nullptr, mmap_alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
        char* end_of_region = given_ptr + mmap_alloc_size;

        // provide
        size_t offset_from_alignment = reinterpret_cast<size_t>(given_ptr) & (ALLOCATION_SIZE - 1);
        char* aligned_ptr = given_ptr;
        size_t unneeded_memory_start = 0;
        size_t unneeded_memory_end   = ALLOCATION_SIZE;

        // Calculate proper memory size
        if ( offset_from_alignment != IDEAL_OFFSET) {
            // align
            aligned_ptr = given_ptr - offset_from_alignment + ALLOCATION_SIZE;
            unneeded_memory_start = static_cast<size_t>(aligned_ptr - given_ptr);
            unneeded_memory_end   = end_of_region - aligned_ptr - ALLOCATION_SIZE;

            munmap(given_ptr, unneeded_memory_start);
        }

        // Unmap unneeded assigned memory
        munmap(aligned_ptr + ALLOCATION_SIZE, unneeded_memory_end);

        ChunkHeader* header = reinterpret_cast<ChunkHeader*>( aligned_ptr - sizeof(ChunkHeader) );
        header->initialize(StorageType::CONTINUOUS);
        //header->info->status->type = StorageType::CONTINUOUS;
        // memset bitmap to initialize status
        header->reset();

        return reinterpret_cast<void*>(aligned_ptr);
    }

    // Just allocate with StorageType padded in front
    // size must be larger than 128MB
    // TODO: it must be aligned to work AT ALL
    void* allocateLarge(size_t size)
    {
        assert(size % LINUX_PAGE_SIZE == 0);
        assert(size > (1 << 27));
        char* ptr = ::new char[size + sizeof(StorageType)];
        StorageType* type = reinterpret_cast<StorageType*>(ptr);
        ptr += sizeof(StorageType);
        *type = StorageType::LARGE;

        return ptr;
    }

    void* allocatePages(size_t /*size*/, void* /*chunk_location*/)
    {
        return nullptr;
    }

    void* allocate(size_t size) override
    {
        if (size > ALLOCATION_SIZE) {
            return allocateLarge(size);
        }
        else {
            // TODO: concurrency
            if (m_current_chunk == nullptr)
                m_current_chunk = allocateContinuous();
            return allocatePages(size, m_current_chunk);
        }
    }

    void deallocate(void* const /*ptr*/) override
    {

    }

    void deallocateAll()
    {

    }

    void *allocate(abstract_memory_manager *const /*manager*/, size_t /*size*/) override
    {
         return nullptr;
    }

    void deallocate(abstract_memory_manager *const /*manager*/, void *const /*ptr*/) override
    {

    }

    void * reallocate(abstract_memory_manager * const /*manager*/, void * /*ptr*/, size_t /*size*/) override
    {
        return nullptr;
    }

    void * reallocate(void* ptr, size_t size) override
    {
        return reallocate_front(ptr, size);
    }

    void * reallocate_front(void* /*ptr*/, size_t /*size*/)
    {
        return nullptr;
    }

    void * reallocate_back(void* /*ptr*/, size_t /*size*/)
    {
        return nullptr;
    }

    void handle_error() override
    {

    }

private:
    mmap_memory_manager() : m_current_chunk(nullptr) {}

    // used for direct allocation calls
    void* m_current_chunk;
    std::mutex m_Mutex;

};



} //namespace morphstore

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
