#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H

#include <mutex>
#include <mman.h>

#include <assert>

namespace morphstore {

const size_t LINUX_PAGE_SIZE = 1 << 12;
const size_t DB_PAGE_SIZE = 1 << 14;
const size_t ALLOCATION_SIZE = 128_MB;
const size_t ALLOCATION_OFFSET = 27;

enum class StorageType : public uint64_t {
    CONTINUOUS;
    LARGE;
};

class AllocationStatus {
public:
    AllocationStatus() : next(nullptr), curr_offset(0), StorageType(StorageType::CONTINUOUS) {}
    ChunkHeader* next;
    uint64_t curr_offset;
    StorageType type;
    std::mutex sema;
};

class InfoHeader {
public:
    char unused[LINUX_PAGE_SIZE - sizeof(AllocationStatus)];
    AllocationStatus status;
};
static_assert(sizeof(InfoHeader) == 4096);

class ChunkHeader {
    //place holder
    //size must be aligned to linux page size
    // Bitfield is used as follows: one entry for each page is 2 bit
    // UNUSED  = 00
    // DEALLOC = 01
    // START   = 10
    // CONT    = 11
public:
    static const uint64_t ALLOC_BITS = 2;
    static const uint8_t UNUSED = 0;
    void reset()
    {
        memset(bitmap, UNUSED, LINUX_PAGE_SIZE);
    }

    void* getCurrentOffset()
    {
        char* chunkLocation = reinterpret_cast<char*>(this) + sizeof(ChunkHeader);
        return chunkLocation + reinterpret_cast<char*>(info->status->curr_offset);
    }

    // TODO: implement atomic variant to enable lock-free allocation
    void* forwardOffset(size_t size)
    {
        char* address = getCurrentOffset();
        address += size;
        char* chunkLocation = reinterpret_cast<char*>(this) + sizeof(ChunkHeader);
        info->status->curr_offset = address;
    }

    void setAllocated(void* address, size_t size)
    {
        char* location_this = reinterpret_cast<char*>(this);
        char* location_in_chunk = reinterpret_cast<char*>(address) - location_this - sizeof(ChunkHeader);

        long needed_blocks = size / static_cast<long>(DB_PAGE_SIZE) + 1;

        // Calculate precise location in bitmap
        uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
        //uint64_t byte_pos_in_bitmap = bit_offset >> 3;
        uint64_t word_aligned_pos_in_bitmap = bit_offset >> 6;
        uint64_t bit_offset_in_word = bit_offset & 0x3fl;

        long allocate_in_word = 32 - bit_offset_in_word / 2;
        
        uint64_t* word_in_bitmap = reinterpret_cast<uint64_t>(bitmap + word_aligned_pos_in_bitmap);
        uint64_t state = getAllocBits64(bit_offset_in_word, needed_blocks);

        *word_in_bitmap |= state;
        needed_blocks -= allocate_in_word;
        // Move forward in bitmap by one word
        word_in_bitmap = reinterpret_cast<char*>(word_in_bitmap) + sizeof(uint64_t);
        while (needed_blocks > 0) {
             state = getAllocBits64(0, needed_blocks, false);
             *word_in_bitmap = state;
             needed_blocks -= sizeof(uint64_t);
        }
    }

    void setDeallocated(void* start)
    {
        // TODO: just do it

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
            return (~(0l) - (1 << 63));

        // TODO: optimize, dunno if compiler optimizes
        while (i < 64/ALLOC_BITS && count_blocks > 0) {
            state |= (i == startOffset && isStart ? 0b10 : 0b11) << (64 - i*ALLOC_BITS);
            --count_blocks;
            ++i; 
        }

        return state;
    }
};
static_assert(sizeof(ChunkHeader) == 8192);

const size_t HEAD_STRUCT = sizeof(ChunkHeader);
const size_t IDEAL_OFFSET = ALLOCATION_SIZE - HEAD_STRUCT;

class mmap_memory_manager: public abstract_memory_manager {
public:
    mmap_memory_manager& getInstance()
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
        char* given_ptr = mmap(nullptr, mmap_alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        char* end_of_region = given_ptr + mmap_alloc_size;

        // provide
        char* offset_from_alignment = given_ptr & (ALLOCATION_SIZE - 1);
        char* aligned_ptr = given_ptr;
        size_t unneeded_memory_start = 0;
        size_t unneeded_memory_end   = ALLOCATION_SIZE;

        // Calculate proper memory size
        if ( offset_from_aligment != IDEAL_OFFSET) {
            aligned_ptr = ptr - offset_from_alignment + ALLOCATION_SIZE;
            unneeded_memory_start = reinterpret_cast<size_t>(aligned_ptr - given_ptr);
            unneeded_memory_end   = end_of_region - aligned_ptr - ALLOCATION_SIZE;

            munmap(given_ptr, unneeded_memory_start);
        }

        // Unmap unneeded assigned memory
        munmap(aligned_ptr + ALLOCATION_SIZE, unneeded_memory_end);

        ChunkHeader* header = reinterpret_cast<ChunkHeader*>( aligned_ptr - sizeof(ChunkHeader) );
        (*(header->info->status))();
        //header->info->status->type = StorageType::CONTINUOUS;
        // memset bitmap to initialize status
        header->reset();

        return reinterpret_cast<void*>(aligned_ptr);
    }

    // Just allocate with StorageType padded in front
    // size must be larger than 128MB
    void* allocateLarge(size_t size)
    {
        assert(size % LINUX_PAGE_SIZE == 0);
        assert(size > (1 << 27));
        char* ptr = ::new char[size + sizeof(StorageType)];
        StorageType* type = reinterpret_cast<StorageType*>(ptr);
        ptr += sizeof(StorageType);
        type = StorageType::LARGE;

        return ptr;
    }

    void* allocatePages(size_t size)
    {

        return nullptr;
    }

    void* allocate(size_t size) override
    {
        if (size > 128_MB)
            return allocateLarge(size);
        else
            return allocatePages(size);
    }

    void deallocate(void* const ptr) override
    {

    }

    void * reallocate(void* ptr, size_t size) override
    {
        return reallocate_front(ptr, size);
    }

    void * reallocate_front(void* ptr, size_t size) override
    {
        return nullptr;
    }

    void * reallocate_back(void* ptr, size_t size)
    {
        return nullptr;
    }

    void handle_error() override
    {

    }

private:
    mmap_memory_manager() : m_Instance(nullptr) {}

    static mmap_memory_manager* m_Instance;

    std::mutex m_Mutex;

};

} //namespace morphstore

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
