#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H

//#include <core/memory/global/mm_hooks.h>
//#include <core/utils/logger.h>

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
const size_t DB_PAGE_OFFSET = 15;
const size_t DB_PAGE_SIZE = 1 << DB_PAGE_OFFSET;
const size_t ALLOCATION_SIZE = 1l << 27;
const size_t ALLOCATION_OFFSET = 27;

enum StorageType : uint64_t {
    CONTINUOUS,
    LARGE,
    PAGE
};

struct ObjectInfo {
    ObjectInfo(StorageType allocType, size_t allocSize)
    {
        type = static_cast<uint64_t>(allocType);
        size = allocSize;
    }

    uint64_t type : 8;
    uint64_t size : 56;
};

class ChunkHeader;

class AllocationStatus {
public:
    AllocationStatus(size_t alloc_size) : m_next(nullptr), m_curr_offset(0), m_info((StorageType::CONTINUOUS), alloc_size) {}
    ChunkHeader* m_next;
    uint64_t m_curr_offset;
    std::mutex m_sema;
    ObjectInfo m_info;
};

class InfoHeader {
public:
    char unused[LINUX_PAGE_SIZE - sizeof(AllocationStatus)];
    AllocationStatus status;
};
static_assert(sizeof(InfoHeader) == 4096, "Assuming info header should be page size'd");

class ChunkHeader {
public:
    static const uint64_t ALLOC_BITS = 2;
    //size must be aligned to linux page size
    // Bitfield is used as follows: one entry for each page is 2 bit
    static const uint8_t UNUSED = 0b00;
    static const uint8_t UNMAP  = 0b01;
    static const uint8_t START  = 0b10;
    static const uint8_t CONT   = 0b11;

    void initialize(StorageType type)
    {
        m_info.status.m_next = nullptr;
        m_info.status.m_curr_offset = 0;
        m_info.status.m_info.type = (type);
        m_info.status.m_info.size = ALLOCATION_SIZE;
    }

    void vol_memset(volatile void *s, char c, size_t n)
    {
        volatile char *p = reinterpret_cast<volatile char*>(s);
        for (; n>0; --n) {
            *p = c;
            ++p;
        }
    }

    void reset()
    {
        vol_memset(bitmap, UNUSED, LINUX_PAGE_SIZE);
    }

    void* getCurrentOffset()
    {
        size_t chunkLocation = reinterpret_cast<size_t>(this) + sizeof(ChunkHeader);
        return reinterpret_cast<void*>(chunkLocation + reinterpret_cast<size_t>(m_info.status.m_curr_offset));
    }

    // TODO: implement atomic variant to enable lock-free allocation
    // Not used yet
    void forwardOffset(size_t size)
    {
        char* address = reinterpret_cast<char*>(getCurrentOffset());
        address += size;
        //char* chunkLocation = reinterpret_cast<char*>(this) + sizeof(ChunkHeader);
        m_info.status.m_curr_offset = reinterpret_cast<uint64_t>(address);
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
        uint64_t word_aligned_pos_in_bitmap = bit_offset >> 6;
        uint64_t bit_offset_in_word = bit_offset & 0x3fl;

        long allocate_in_word = 32 - bit_offset_in_word / 2;
        
        volatile uint64_t* word_in_bitmap = reinterpret_cast<volatile uint64_t*>(getBitmapAddress() + sizeof(uint64_t) * word_aligned_pos_in_bitmap);
        ////trace("Bit offset in word: ", bit_offset_in_word);
        ////trace("word aligned pos in bitmap: ", word_aligned_pos_in_bitmap);
        uint64_t state = getAllocBits64(bit_offset_in_word, needed_blocks);

        *word_in_bitmap |= state;
        needed_blocks -= allocate_in_word;
        // Move forward in bitmap by one word
        word_in_bitmap = reinterpret_cast<volatile uint64_t*>(reinterpret_cast<volatile uint64_t>(word_in_bitmap) + sizeof(uint64_t));
        while (needed_blocks > 0) {
             state = getAllocBits64(0, needed_blocks, false);
             *word_in_bitmap = state;
             needed_blocks -= sizeof(uint64_t) * 8 / ALLOC_BITS;
             word_in_bitmap = reinterpret_cast<volatile uint64_t*>(reinterpret_cast<uint64_t>(word_in_bitmap) + sizeof(uint64_t));
        }

        //dumpBitmap();
    }

    void setDeallocated(void* address)
    {
        dumpBitmap();
        uint64_t location_chunk = reinterpret_cast<uint64_t>(address) & ~(ALLOCATION_SIZE - 1);
        uint64_t location_in_chunk = reinterpret_cast<uint64_t>(address) - (location_chunk );
        location_in_chunk = location_in_chunk / DB_PAGE_SIZE;

        // Calculate precise location in bitmap
        //trace("got location ", location_in_chunk);
        uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
        uint64_t word_aligned_pos_in_bitmap = bit_offset >> 6;
        uint64_t bit_offset_in_word = bit_offset & 0x3fl;

        // Find out length of allocated word
        // Mask which starts with 1s at the first allocation bits
        volatile uint64_t* word_in_bitmap = reinterpret_cast<volatile uint64_t*>(getBitmapAddress() + word_aligned_pos_in_bitmap);
        (void) word_in_bitmap;
        //trace("bitmap: ", std::hex, getBitmapAddress(), ", this: ", this);
        //trace("got addr ", reinterpret_cast<uint64_t>(word_in_bitmap), " in bitmap ", std::hex, getBitmapAddress(), " with value ", *word_in_bitmap);
        
        // probe for start bits
        volatile uint64_t* start_word = reinterpret_cast<uint64_t*>(getBitmapAddress() + word_aligned_pos_in_bitmap * sizeof(uint64_t));
        uint64_t start_offset = (bit_offset_in_word);
        //trace("start offset: ", start_offset);
        assert( (*start_word >> (62 - start_offset) & 0b11) == 0b10 );

        //trace("*word is ", std::hex, *start_word);
        *start_word = *start_word & ~(0b11l << (62 - start_offset));
        //trace("*word is now ", std::hex, *start_word);

        // increment due to start
        bit_offset_in_word += ALLOC_BITS;

        bool endFound = false;
        //TODO: increment by word
        for (uint64_t i = bit_offset_in_word + ALLOC_BITS; !endFound; i += ALLOC_BITS) {
            volatile uint64_t* word = reinterpret_cast<uint64_t*>(getBitmapAddress() + (word_aligned_pos_in_bitmap + (i>>6)) * sizeof(uint64_t));
            uint64_t offset = i & 0x3fl;
            if ( ((*word >> (64 - offset)) & 0b11) == 0b11 ) {
                //trace("*word is ", std::hex, *word);
                *word = *word & ~(0l + (0b11 << offset));
                //trace("*word is now ", std::hex, *word);
            }
            else {
                endFound = true;
            }
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
        volatile uint64_t* loc = reinterpret_cast<volatile uint64_t*>(bitmap);
        const uint64_t slots_needed = (size >> DB_PAGE_OFFSET) + ( ((size % DB_PAGE_SIZE) == 0) ? 0 : 1);

        uint64_t continuousPageCounter = 0;
        //bool runningContinuous = false;
        uint64_t bit_start = 0;

        const size_t end_of_allocation_bitmap = reinterpret_cast<size_t>(bitmap) + (ALLOCATION_SIZE / DB_PAGE_SIZE * ALLOC_BITS / 8 /*Bits per byte*/);
        ////trace( "End of bitmap is on ", std::hex, end_of_allocation_bitmap);

        while (reinterpret_cast<uint64_t>(loc) < end_of_allocation_bitmap) {
            //First check if space is not full
            if (*loc < ~(0x4ul << 60)) {
                ////trace( std::hex, *loc, " allocation map for 8 bytes on ", loc);
                uint64_t bit_offset = ALLOC_BITS;

                // check within one word
                while (bit_offset <= 64) {
                    uint8_t bits = (*loc >> (64 - bit_offset)) & 0b11ul;
                    ////trace( "Bits are ", std::hex, static_cast<uint32_t>(bits));
                    // space is available
                    if (bits == 0) {
                        bit_start = (reinterpret_cast<uint64_t>(loc) - reinterpret_cast<uint64_t>(bitmap)) * 8 + bit_offset - ALLOC_BITS;
                        //runningContinuous = true;
                        //uint64_t bit_start = reinterpret_cast<char*>(loc) - bitmap + bit_offset;
                        ++continuousPageCounter;
                        if (continuousPageCounter >= slots_needed) {
                            ////trace( "Returning blocks with offset ", bit_start/2, ", this ", this, " chunkheader ", std::hex, sizeof(ChunkHeader));
                            void* addr = reinterpret_cast<void*>(
                                    reinterpret_cast<uint64_t>(this) + sizeof(ChunkHeader) + DB_PAGE_SIZE * (bit_start / 2));
                            ////trace( "Returning address ", addr);
                            assert(reinterpret_cast<uint64_t>(this) + sizeof(ChunkHeader) + ALLOCATION_SIZE > reinterpret_cast<uint64_t>(addr));
                            return addr; 
                        }
                    }
                    else {
                        bit_start = 0;
                        continuousPageCounter = 0;
                    }
                    bit_offset += 2;
                }
            }
            else {
                //runningContinuous = false;
                bit_start = 0;
                continuousPageCounter = 0;
                //trace("set bit_start and continuous page counter to 0");
            }
            loc = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(loc) + sizeof(uint64_t));
            ////trace( "loc moved forward to ", loc);
        }

        return nullptr;
    }

    inline uint64_t getBitmapAddress()
    {
        return reinterpret_cast<uint64_t>(bitmap);
    }

    volatile void* getBitmapEnd()
    {
        return reinterpret_cast<volatile void*>( getBitmapAddress() + ALLOCATION_SIZE / DB_PAGE_SIZE * ALLOC_BITS / 8);
    }

    volatile char bitmap[LINUX_PAGE_SIZE];
    InfoHeader m_info;

//private:
    inline uint64_t getAllocBits64(uint8_t startOffset, uint32_t count_blocks, bool isStart = true)
    {
        // Increment since we start from the first two indicated bits
        startOffset += ALLOC_BITS;
        uint8_t i = startOffset;
        uint64_t state = 0;

        // Start is marked directly at first two MSB
        //if (startOffset == 0 && isStart)
        //    return (~(0l) - (1l << 63));

        // TODO: optimize, dunno if compiler optimizes
        while (i <= 64 && count_blocks > 0) {
            state = state | ((i == startOffset && isStart ? 0b10ul : 0b11ul) << (64 - i));
            --count_blocks;
            i+=ALLOC_BITS; 
            ////trace(std::hex, state);
        }

        return state;
    }

    void dumpBitmap()
    {
        uint64_t bitmap_pos = getBitmapAddress();
        //trace( "Dumping bitmap on ", std::hex, bitmap_pos, ", this being ", this);
        uint64_t bitmap_end = bitmap_pos + LINUX_PAGE_SIZE;
        uint32_t cycles = 0;

        while (bitmap_pos < bitmap_end) {
            uint64_t* bitmap_value = reinterpret_cast<uint64_t*>(bitmap_pos);
            //unused //warning
            (void) bitmap_value;
            //trace( std::hex, *bitmap_value, " ");
            bitmap_pos += sizeof(uint64_t);
            ++cycles;

            //debug
            if (cycles > 4)
                return;
        }
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
        /*if (m_Instance == nullptr) {
            mmap_memory_manager* ptr = reinterpret_cast<mmap_memory_manager*>(
                    mmap(nullptr, LINUX_PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
            m_Instance = new (ptr) mmap_memory_manager();
        }*/
        static mmap_memory_manager instance;
        return instance;
    }
    ~mmap_memory_manager() {}

    /**
     * @brief use for continuous memory allocation
     */
    void* allocateContinuous()
    {
        // need at least 2*alloc_size-1 for guaranteed alignment
        //trace("[MMAP_MM] Allocation of continuous chunk requested");
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
            unneeded_memory_start = static_cast<size_t>(aligned_ptr - given_ptr - sizeof(ChunkHeader));
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

        //trace("[MMAP_MM] returning ", reinterpret_cast<void*>(aligned_ptr), " as new chunk");
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

    void* allocatePages(size_t size, void* chunk_location)
    {
        ////trace( "Page allocation request for size ", size, " on location ", chunk_location);
        if (size < DB_PAGE_SIZE) {
            //warn( "[MMAP_MM] Request for too small size ", std::hex, size, " has been made, rounding up...");
            size = DB_PAGE_SIZE;
        }

        assert((reinterpret_cast<uint64_t>(chunk_location) & (ALLOCATION_SIZE-1)) == 0);
        ChunkHeader* header = reinterpret_cast<ChunkHeader*>(reinterpret_cast<uint64_t>(chunk_location) - sizeof(ChunkHeader));
        void* ptr = header->findNextAllocatableSlot(size);
        ////trace( "header on ", header, " found next allocatable slot as ", ptr);
        if (ptr == nullptr)
            return nullptr;
        header->setAllocated(ptr, size);

        return ptr;
    }

    void* allocate(size_t size, void* chunk_location) 
    {
        if (size > ALLOCATION_SIZE) {
            //warn("Allocating large object of size");
            return allocateLarge(size);
        }
        else {
            // TODO: concurrency
            if (chunk_location != nullptr) {
                ////trace("Allocating pages with chunk ", chunk_location);
                return allocatePages(size, chunk_location);
            }
            else if (m_current_chunk == nullptr) {
                ////trace("Using allocator specific chunk on ", m_current_chunk);
                m_current_chunk = allocateContinuous();
            }

            return allocatePages(size, m_current_chunk);
        }
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

    void deallocate(void* const ptr) override
    {
        uint64_t chunk_ptr = reinterpret_cast<uint64_t>(ptr) & ~(ALLOCATION_SIZE - 1);
        ChunkHeader* header = reinterpret_cast<ChunkHeader*>( chunk_ptr - sizeof(ChunkHeader) );
        //trace("deallocating ", ptr, " with header on ", header);
        header->setDeallocated(ptr);
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
