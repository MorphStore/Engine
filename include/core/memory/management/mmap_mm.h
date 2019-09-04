#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
#define MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H


#ifndef MORPHSTORE_CORE_MEMORY_MANAGEMENT_ABSTRACT_MM_H
#  error "Abstract memory manager ( management/abstract_mm.h ) has to be included before mmap memory manager."
#endif

#ifndef MORPHSTORE_CORE_MEMORY_GLOBAL_MM_HOOKS_H
#  error "Memory Hooks ( global/mm_hooks.h ) has to be included before general memory manager."
#endif

#include <mutex>
#include <sys/mman.h>
#include <string.h>

#include <cassert>

#define USE_FREEMAP
//#define USE_HUGE_TLB
//#define USE_VECTOR

namespace morphstore {

const size_t LINUX_PAGE_SIZE = 1 << 12;
#ifdef CUSTOM_PAGE_OFFSET
const size_t DB_PAGE_OFFSET = CUSTOM_PAGE_OFFSET > 12 ? CUSTOM_PAGE_OFFSET : 12;
#else
const size_t DB_PAGE_OFFSET = 15;
#endif

#ifdef CUSTOM_ALLOCATION_OFFSET
const size_t ALLOCATION_OFFSET = CUSTOM_ALLOCATION_OFFSET;
#else
const size_t ALLOCATION_OFFSET = 27;
#endif

const size_t DB_PAGE_SIZE = 1ul << DB_PAGE_OFFSET;
const size_t ALLOCATION_SIZE = 1ul << ALLOCATION_OFFSET;

class mmap_memory_manager : public abstract_memory_manager {
public:
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

        size_t getSize()
        {
            return static_cast<size_t>(size);
        }

        uint64_t type : 8;
        uint64_t size : 56;
    };

    struct MemNode {
        MemNode* prev;
        MemNode* next;
    };

    struct MemPair {
        MemPair() : memloc(0), count(0) {}
        size_t memloc : 50;
        size_t count : 14;

        MemNode* getFirstMemNode()
        {
            if (memloc == 0)
                return nullptr;
            return reinterpret_cast<MemNode*>( memloc << 14 );
        }

        size_t getCount()
        {
            return count;
        }

        MemNode* popFirstMemNode()
        {
            if (memloc == 0)
                return nullptr;

            --count;
            MemNode* node = reinterpret_cast<MemNode*>( memloc << 14);
            if (node->next == nullptr) {
                memloc = 0;
            }
            else {
                MemNode* newFirst = node->next;
                memloc = reinterpret_cast<size_t>( node->next ) >> 14 ;
                newFirst->prev = node->prev;
            } 
            return node;
        }

        void appendMemNode(void* ptr)
        {
            MemNode* node = reinterpret_cast<MemNode*>(ptr);
            node->next = nullptr;
            node->prev = nullptr;

            if (memloc == 0) {
                node->next = nullptr;
                node->prev = node;

                memloc = reinterpret_cast<size_t>(ptr) >> 14;
                ++count;
                return;
            }

            ++count;

            MemNode* first = reinterpret_cast<MemNode*>( memloc << 14 );
            MemNode* endNode = first->prev;

            endNode->next = node;
            node->prev = endNode;
            first->prev = node;
        }
    };

    template<
        size_t ALLOCATION_SIZE,
        size_t DB_PAGE_SIZE
            >
    struct ListMap {
        MemPair pairs[ALLOCATION_SIZE / DB_PAGE_SIZE];

        void init()
        {
            for (uint64_t i = ALLOCATION_SIZE / DB_PAGE_SIZE; i > 0; --i) {
                new (&pairs[i - 1]) MemPair();
            }
        }

        bool hasTooManyEntries(size_t size)
        {
            size_t aligned = getAlignedSize(size);
            u_int64_t index = getMemNodeEntry(aligned);

            size_t count = pairs[index].getCount();
            //TODO: make dependent on size
            if (count > (size < (1 << 24) ? 10 : 5 ))
                return true;
            else
                return false;
        }

        void insert(void* memloc, size_t size)
        {
            size_t aligned_size = getAlignedSize(size);

            u_int64_t index = getMemNodeEntry(aligned_size);
            pairs[index].appendMemNode(memloc);
        }

        void* getMemory(size_t requestedSize)
        {
            size_t align = getAlignedSize(requestedSize);
            size_t index = getMemNodeEntry(align);
            void* loc = pairs[index].popFirstMemNode();
            return loc;
        }

        inline size_t getAlignedSize(size_t size)
        {
            return ( size % DB_PAGE_SIZE == 0 ? size : (size & ~( DB_PAGE_SIZE - 1) ) + DB_PAGE_SIZE );
        }

        inline size_t getMemNodeEntry(size_t size)
        {
            return ( size / DB_PAGE_SIZE + ( size % DB_PAGE_SIZE == 0 ? 0 : 1 ) - 1 );
        }

        inline size_t getMemNodeSize(size_t digit)
        {
            return ( digit + 1 ) * DB_PAGE_SIZE;
        }
    };
public:
    class ChunkHeader;

    class AllocationStatus {
    public:
        AllocationStatus(size_t alloc_size) : m_next(nullptr), m_prev(nullptr), m_refcount(0), m_totalRegions(0), m_info((StorageType::CONTINUOUS), alloc_size) {}
        void* m_next;
        void* m_prev;
        uint32_t m_refcount;
        uint64_t m_totalRegions;
        std::mutex m_sema;
        ObjectInfo m_info;
        uint64_t* lastBitmapLocation;
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

        static ChunkHeader* getChunkHeader(void* address)
        {
            return reinterpret_cast<ChunkHeader*>( (reinterpret_cast<uint64_t>(address) & ~(ALLOCATION_SIZE - 1)) - sizeof(ChunkHeader));
        }

        static void* getChunkPointer(ChunkHeader* header)
        {
            return reinterpret_cast<void*>(reinterpret_cast<uint64_t>(header) + sizeof(ChunkHeader));
        }

        void initialize(StorageType type)
        {
            m_info.status.m_next = nullptr;
            m_info.status.m_prev = nullptr;
            m_info.status.m_totalRegions = 0;
            m_info.status.m_info.type = (type);
            m_info.status.m_info.size = ALLOCATION_SIZE;
            m_info.status.lastBitmapLocation = reinterpret_cast<uint64_t*>(bitmap);
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


        void* getNext()
        {
            return m_info.status.m_next;
        }

        void setNext(void* nextChunk)
        {
            m_info.status.m_next = nextChunk;
        }

        void* getPrev()
        {
            return m_info.status.m_prev;
        }

        void setPrev(void* prevChunk)
        {
            m_info.status.m_prev = prevChunk;
        }

        void setAllocated(void* address, size_t size)
        {
            //TODO: need to lock chunk to increment regions, or do atomic
            m_info.status.m_totalRegions++; 
            char* location_this = reinterpret_cast<char*>(this);
            uint64_t location_in_chunk = reinterpret_cast<uint64_t>(address) - reinterpret_cast<uint64_t>(location_this + sizeof(ChunkHeader));
            location_in_chunk = location_in_chunk >> DB_PAGE_OFFSET;

            long needed_blocks = size >> static_cast<long>(DB_PAGE_OFFSET);
            if (size % DB_PAGE_SIZE != 0)
                ++needed_blocks;
            trace("Setting allocated on ", address, " for size ", size, " needing ", needed_blocks, " pages");

            // Calculate precise location in bitmap
            uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
            //uint64_t byte_pos_in_bitmap = bit_offset >> 3;
            uint64_t word_aligned_pos_in_bitmap = bit_offset >> 6;
            uint64_t bit_offset_in_word = bit_offset & 0x3fl;

            long allocate_in_word = 32 - bit_offset_in_word / 2;
            
            uint64_t* word_in_bitmap = reinterpret_cast<uint64_t*>(getBitmapAddress() + sizeof(uint64_t) * word_aligned_pos_in_bitmap);
            uint64_t state = getAllocBits64(bit_offset_in_word, needed_blocks);

            trace("Word in bitmap is now ", *word_in_bitmap, ", now allocating ", needed_blocks, " needed blocks");
            *word_in_bitmap |= state;
            needed_blocks -= allocate_in_word;
            trace("Word in bitmap is now ", *word_in_bitmap);
            // Move forward in bitmap by one word
            word_in_bitmap = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(word_in_bitmap) + sizeof(uint64_t));
            while (needed_blocks > 0) {
                 state = getAllocBits64(0, needed_blocks, false);
                 trace("Received further state in bitmap as ", std::hex, state, ", needed blocks is still ", needed_blocks);
                 *word_in_bitmap = state;
                 needed_blocks -= sizeof(uint64_t) * 8 / ALLOC_BITS;
                 trace("Word in bitmap is now ", *word_in_bitmap, " on ", word_in_bitmap);
                 word_in_bitmap = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(word_in_bitmap) + sizeof(uint64_t));
            }

            //// Checks for non-release
            //uint64_t last_page_offset = (size % DB_PAGE_SIZE == 0) ? size - DB_PAGE_SIZE : size - (size % DB_PAGE_SIZE);
            //assert(!isAllocatable( reinterpret_cast<void*>( reinterpret_cast<uint64_t>(address)), DB_PAGE_SIZE ) );
            //assert(!isAllocatable( reinterpret_cast<void*>( reinterpret_cast<uint64_t>(address) + last_page_offset ), DB_PAGE_SIZE ) );
            
        }

        void setDeallocated(void* address)
        {
            //dumpBitmap();
            trace("Deallocating ", address);
            uint64_t location_chunk = reinterpret_cast<uint64_t>(address) & ~(ALLOCATION_SIZE - 1);
            uint64_t location_in_chunk = reinterpret_cast<uint64_t>(address) - (location_chunk );
            location_in_chunk = location_in_chunk >> DB_PAGE_OFFSET;

            // Calculate precise location in bitmap
            uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
            uint64_t word_aligned_pos_in_bitmap = bit_offset >> 6;
            uint64_t bit_offset_in_word = bit_offset & 0x3fl;

            // Find out length of allocated word
            // Mask which starts with 1s at the first allocation bits
            //uint64_t* word_in_bitmap = reinterpret_cast<uint64_t*>(getBitmapAddress() + word_aligned_pos_in_bitmap);
            //(void) word_in_bitmap;
            
            // probe for start bits
            uint64_t* start_word = reinterpret_cast<uint64_t*>(getBitmapAddress() + word_aligned_pos_in_bitmap * sizeof(uint64_t));
            uint64_t start_offset = (bit_offset_in_word);

            if( (*start_word >> (62 - start_offset) & 0b11ul) != 0b10ul )
                warn("Address on ", address, " was indicated as not have been allocated, word in allocation bitmap is ", std::hex, *start_word);

            *start_word = *start_word & ~(0b11ul << (62 - start_offset));
            // increment due to start
            bit_offset_in_word += ALLOC_BITS;

            bool endFound = false;
            int dealloced_page_counter = 1;

            //TODO: increment by word
            for (uint64_t i = bit_offset_in_word; !endFound; i += ALLOC_BITS) {
                uint64_t* word = reinterpret_cast<uint64_t*>(getBitmapAddress() + (word_aligned_pos_in_bitmap + (i>>6)) * sizeof(uint64_t));
                uint64_t offset = (i & 0x3ful) + ALLOC_BITS;
                trace("Word was ", std::hex, *word);

                uint64_t bit_shift_res = ((*word >> (64 - offset)) & 0b11ul);
                if ( bit_shift_res == 0b11ul ) {
                    //uint64_t former_word = *word;
                    *word = *word & ~(0ul + (0b11ul << (64 - offset)));
                    trace("Negative part is ", ~(0ul + (0b11ul << (64 - offset))));
                    trace("Word is now ", std::hex, *word, " as ", bit_shift_res, " was negated at ", offset);
                    //assert(former_word != *word);
                    dealloced_page_counter++;
                }
                else {
                    return;
                    trace("Word is now ", std::hex, *word, ", end found, dealloced ", dealloced_page_counter, " pages");
                    endFound = true;
                }
            }

            //TODO: lock
            m_info.status.m_totalRegions--;

            //if (m_info.status.m_totalRegions < 10 )
                //std::cout << "Total number of used regions " << m_info.status.m_totalRegions << std::endl;

        }

        void setDeallocatedEnd(void* address, size_t original_size, size_t new_size)
        {
            trace("Called on ", address, " for original size ", original_size, ", and new size ", new_size);
            size_t blocks_original = (original_size >> DB_PAGE_OFFSET) + (original_size % DB_PAGE_SIZE == 0 ? 0 : 1);
            size_t blocks_new = (new_size >> DB_PAGE_OFFSET) + (new_size % DB_PAGE_SIZE == 0 ? 0 : 1);

            if (blocks_new == blocks_original)
                return;

            uint64_t bit_offset = ALLOC_BITS * blocks_new;
            uint64_t bits_to_be_set_to_zero = (blocks_original - blocks_new) * ALLOC_BITS;
            uint64_t word_aligned_pos_in_bitmap = reinterpret_cast<uint64_t>(address) + (bit_offset >> 6) - reinterpret_cast<uint64_t>(getChunkPointer(this));
            uint64_t bit_offset_in_word = bit_offset & 0x3fl;

            uint64_t* start_word = reinterpret_cast<uint64_t*>(getBitmapAddress() + word_aligned_pos_in_bitmap * sizeof(uint64_t));
            uint64_t start_offset = (bit_offset_in_word);

            // set first ALLOC_BITS to zero
            *start_word = *start_word & ~(0b11l << (62 - start_offset));
            // increment due to start
            bit_offset_in_word += ALLOC_BITS;

            for (uint64_t i = bit_offset_in_word; bits_to_be_set_to_zero > 0; bits_to_be_set_to_zero -= ALLOC_BITS) {
                uint64_t* word = reinterpret_cast<uint64_t*>(getBitmapAddress() + (word_aligned_pos_in_bitmap + (i>>6)));
                uint64_t offset = (i & 0x3fl) + ALLOC_BITS;
                if ( ((*word >> (64 - offset)) & 0b11) == 0b11 ) {
                    *word = *word & ~(0l + (0b11 << offset));
                }
                else {
                    warn("Area was not allocated");
                }
            }
        }

        // used to decide whether we can reallocate in front or behind a memory location
        bool isAllocatable(void* start, size_t size)
        {
            trace("Called on ", start, " with size ", size);
            char* location_this = reinterpret_cast<char*>(this);
            uint64_t location_in_chunk = reinterpret_cast<uint64_t>(start) - reinterpret_cast<uint64_t>(location_this + sizeof(ChunkHeader));
            if (location_in_chunk + size > ALLOCATION_SIZE)
                throw std::runtime_error("Location is outside of chunk");
            location_in_chunk = location_in_chunk / DB_PAGE_SIZE;

            long needed_blocks = size / static_cast<long>(DB_PAGE_SIZE);
            if (size % DB_PAGE_SIZE != 0)
                ++needed_blocks;

            uint64_t bit_offset = ALLOC_BITS * location_in_chunk;
            uint64_t word_aligned_pos_in_bitmap = bit_offset >> 6;
            uint64_t bit_offset_in_word = bit_offset & 0x3ful;

            for (uint64_t i = bit_offset_in_word; needed_blocks > 0; i += ALLOC_BITS, --needed_blocks) {
                uint64_t* word = reinterpret_cast<uint64_t*>(getBitmapAddress() + (word_aligned_pos_in_bitmap + (i>>6)) * sizeof(uint64_t));
                uint64_t offset = (i & 0x3ful) + ALLOC_BITS;
                trace("Got word ", std::hex, *word, " for offset ", offset);
                if ( ((*word >> (64 - offset)) & 0b11ul) != 0b00ul ) {
                    return false;
                }
            }

            return true;
        }

        void* findNextAllocatableSlot(size_t size)
        {
            //assert(size <= ALLOCATION_SIZE);
            // TODO: replace with strategy pattern
            uint64_t* loc = m_info.status.lastBitmapLocation;
            uint64_t* previous_start = loc;
            //uint64_t* loc = reinterpret_cast<uint64_t*>(bitmap);
            const uint64_t start_of_actual_chunk = reinterpret_cast<uint64_t>(this) + sizeof(ChunkHeader);
            const uint64_t slots_needed = (size >> DB_PAGE_OFFSET) + ( ((size % DB_PAGE_SIZE) == 0) ? 0 : 1);
            trace("Trying to find gap for ", slots_needed, " slots");

            uint64_t continuousPageCounter = 0;
            uint64_t bit_start = 0;

            const size_t end_of_allocation_bitmap = reinterpret_cast<size_t>(bitmap) + ( (ALLOCATION_SIZE >> DB_PAGE_OFFSET) * ALLOC_BITS / 8 /*Bits per byte*/);

            while (reinterpret_cast<uint64_t>(loc) + slots_needed*ALLOC_BITS / 8 < end_of_allocation_bitmap) {
                //First check if space is not full
                if (*loc < ~(0x4ul << 60)) {
                    uint64_t bit_offset = ALLOC_BITS;

                    // check within one word
                    while (bit_offset <= 64) {
                        uint8_t bits = (*loc >> (64 - bit_offset)) & 0b11ul;
                        // space is available
                        if (bits == 0) {
                            bit_start = (reinterpret_cast<uint64_t>(loc) - reinterpret_cast<uint64_t>(bitmap)) * 8 + bit_offset - ALLOC_BITS;
                            if ( ALLOCATION_SIZE / DB_PAGE_SIZE * ALLOC_BITS - bit_start < slots_needed * ALLOC_BITS )
                                return nullptr;
                            trace("Current loc for bit_start is ", loc, " with value ", std::hex, *loc);
                            continuousPageCounter++;
                            do {
                                if (continuousPageCounter >= slots_needed) {
                                    void* addr = reinterpret_cast<void*>(
                                            start_of_actual_chunk + DB_PAGE_SIZE * (bit_start / ALLOC_BITS));

                                    trace("Returning address on ", addr, " as next allocatable slot");
                                    m_info.status.lastBitmapLocation = loc;
                                    return addr; 
                                }

                                bit_offset += ALLOC_BITS;
                                if (bit_offset > 64) {
                                    bit_offset = ALLOC_BITS;
                                    loc = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(loc) + sizeof(uint64_t));
                                }
                                
                                bits = (*loc >> (64 - bit_offset)) & 0b11ul;
                                if (bits == 0)
                                    continuousPageCounter++;
                            } while (bits == 0);

                            trace("Bits ", bits, " on offset ", bit_offset, " were not zero");
                            bit_start = 0;
                            continuousPageCounter = 0;
                        }
                        else {
                            trace("Bits ", bits, " on offset ", bit_offset, " were not zero");
                            bit_start = 0;
                            continuousPageCounter = 0;
                        }
                        bit_offset += ALLOC_BITS;
                    }
                }
                else {
                    //runningContinuous = false;
                    bit_start = 0;
                    continuousPageCounter = 0;
                }
                loc = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(loc) + sizeof(uint64_t));
                trace( "loc moved forward to ", loc);
            }

            const size_t previous_end_in_bitmap = reinterpret_cast<size_t>(previous_start) + ( (ALLOCATION_SIZE >> DB_PAGE_OFFSET) * ALLOC_BITS / 8 /*Bits per byte*/);
            while (reinterpret_cast<uint64_t>(loc) + slots_needed*ALLOC_BITS / 8 < previous_end_in_bitmap) {
                //First check if space is not full
                if (*loc < ~(0x4ul << 60)) {
                    uint64_t bit_offset = ALLOC_BITS;

                    // check within one word
                    while (bit_offset <= 64) {
                        uint8_t bits = (*loc >> (64 - bit_offset)) & 0b11ul;
                        // space is available
                        if (bits == 0) {
                            bit_start = (reinterpret_cast<uint64_t>(loc) - reinterpret_cast<uint64_t>(bitmap)) * 8 + bit_offset - ALLOC_BITS;
                            if ( ALLOCATION_SIZE / DB_PAGE_SIZE * ALLOC_BITS - bit_start < slots_needed * ALLOC_BITS )
                                return nullptr;
                            trace("Current loc for bit_start is ", loc, " with value ", std::hex, *loc);
                            continuousPageCounter++;
                            do {
                                if (continuousPageCounter >= slots_needed) {
                                    void* addr = reinterpret_cast<void*>(
                                            start_of_actual_chunk + DB_PAGE_SIZE * (bit_start / ALLOC_BITS));

                                    /*assert(start_of_actual_chunk + ALLOCATION_SIZE > reinterpret_cast<uint64_t>(addr));
                                    assert(reinterpret_cast<uint64_t>(addr) + size - 1 < start_of_actual_chunk + ALLOCATION_SIZE);
                                    assert(isAllocatable(addr, size));*/

                                    trace("Returning address on ", addr, " as next allocatable slot");
                                    m_info.status.lastBitmapLocation = loc;
                                    return addr; 
                                }

                                bit_offset += ALLOC_BITS;
                                if (bit_offset > 64) {
                                    bit_offset = ALLOC_BITS;
                                    loc = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(loc) + sizeof(uint64_t));
                                }
                                
                                bits = (*loc >> (64 - bit_offset)) & 0b11ul;
                                if (bits == 0)
                                    continuousPageCounter++;
                            } while (bits == 0);

                            trace("Bits ", bits, " on offset ", bit_offset, " were not zero");
                            bit_start = 0;
                            continuousPageCounter = 0;
                        }
                        else {
                            trace("Bits ", bits, " on offset ", bit_offset, " were not zero");
                            bit_start = 0;
                            continuousPageCounter = 0;
                        }
                        bit_offset += ALLOC_BITS;
                    }
                }
                else {
                    //runningContinuous = false;
                    bit_start = 0;
                    continuousPageCounter = 0;
                }
                loc = reinterpret_cast<uint64_t*>(reinterpret_cast<uint64_t>(loc) + sizeof(uint64_t));
                trace( "loc moved forward to ", loc);
            }

            return nullptr;
        }

        bool isReallocatable(void* /*address*/, size_t /*size*/)
        {
            //TODO: implement
            return false;
        }

        inline uint64_t getBitmapAddress()
        {
            return reinterpret_cast<uint64_t>(bitmap);
        }

        void* getBitmapEnd()
        {
            return reinterpret_cast<void*>( getBitmapAddress() + ALLOCATION_SIZE / DB_PAGE_SIZE * ALLOC_BITS / 8);
        }

        inline uint64_t getAllocBits64(uint8_t startOffset, uint32_t count_blocks, bool isStart = true)
        {
            // Increment since we start from the first two indicated bits
            const uint64_t ALLONE = ~0ul;
            const uint64_t STARTONE = ~(1ul << 62);

            if (startOffset == 0 && count_blocks >= 32 && isStart == false)
                return ALLONE;

            if (!isStart) {
                if (count_blocks >= 64u - startOffset)
                    return (ALLONE) >> startOffset;
                else
                    return ((ALLONE) >> startOffset) & ~((1ul << (64 - count_blocks*ALLOC_BITS - startOffset)) -1);
            }
            else {
                if (count_blocks >= 64u - startOffset)
                    return (STARTONE) >> startOffset;
                else
                    return ((STARTONE) >> startOffset) & ~((1ul << (64 - count_blocks*ALLOC_BITS - startOffset)) -1);
            }

            /*startOffset += ALLOC_BITS;
            uint8_t i = startOffset;
            uint64_t state = 0;

            // TODO: optimize, dunno if compiler optimizes
            while (i <= 64 && count_blocks > 0) {
                state = state | ((i == startOffset && isStart ? 0b10ul : 0b11ul) << (64 - i));
                --count_blocks;
                i+=ALLOC_BITS; 
            }

            return state;*/
        }

        void dumpBitmap()
        {
            uint64_t bitmap_pos = getBitmapAddress();
            trace( "Dumping bitmap on ", std::hex, bitmap_pos, ", this being ", this);
            uint64_t bitmap_end = bitmap_pos + LINUX_PAGE_SIZE;
            uint32_t cycles = 0;

            while (bitmap_pos < bitmap_end) {
                uint64_t* bitmap_value = reinterpret_cast<uint64_t*>(bitmap_pos);
                //unused //warning
                (void) bitmap_value;
                //trace( std::hex, *bitmap_value, " ");
                bitmap_pos += sizeof(uint64_t);
                ++cycles;

                if (cycles > 4)
                    return;
            }
        }

    //private:
        char bitmap[LINUX_PAGE_SIZE];
        InfoHeader m_info;

    };
    static_assert(sizeof(ChunkHeader) == 8192, "Code expects ChunkHeader to be two pages long, please refit solution in case that should not hold");

    // Actual implementation of mmap_memory_manager
public:
    static mmap_memory_manager* m_Instance;

#ifdef USE_HUGE_TLB
    const size_t HEAD_STRUCT = 2 << 20;
#else
    const size_t HEAD_STRUCT = sizeof(ChunkHeader);
#endif

    const size_t IDEAL_OFFSET = ALLOCATION_SIZE - HEAD_STRUCT;

    mmap_memory_manager &operator=(mmap_memory_manager const &) = delete;
    virtual ~mmap_memory_manager( void )
    {
             debug("setting mmap mm inactive");
             debug("setting mmap mm inactive - out");
    }

    static inline mmap_memory_manager& getInstance()
    {
        static thread_local mmap_memory_manager instance;
        return instance;
    }

    /**
     * @brief use for continuous memory allocation
     */
    void* allocateContinuous()
    {
        // first check if prealloced chunks are available
        if (m_next_free_chunk != nullptr) {
            void* res = m_next_free_chunk;
            ChunkHeader* header = ChunkHeader::getChunkHeader(m_next_free_chunk);

            m_next_free_chunk = header->getNext();
            if (m_next_free_chunk != nullptr)
                ChunkHeader::getChunkHeader(m_next_free_chunk)->setPrev(nullptr);

            // set back to init;
            header->setNext(nullptr);

            return res;
        }


        // need at least 2*alloc_size-1 for guaranteed alignment
        const size_t mmap_alloc_size = 2 * ALLOCATION_SIZE + HEAD_STRUCT;
#ifdef USE_HUGE_TLB
        char* given_ptr = reinterpret_cast<char*>(mmap(nullptr, mmap_alloc_size, PROT_READ | PROT_WRITE, MAP_HUGETLB | MAP_POPULATE | MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
#else
        char* given_ptr = reinterpret_cast<char*>(mmap(nullptr, mmap_alloc_size, PROT_READ | PROT_WRITE, MAP_POPULATE | MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
#endif
        if (reinterpret_cast<void*>(given_ptr) == reinterpret_cast<void*>(~0l)) {
            //Out of memory...
            throw std::runtime_error("Allocation of memory failed, OOM...");
            return nullptr;
        }
        char* end_of_region = given_ptr + mmap_alloc_size;

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

        //assert(reinterpret_cast<void*>(aligned_ptr) != nullptr);
        ChunkHeader* header = reinterpret_cast<ChunkHeader*>( aligned_ptr - sizeof(ChunkHeader) );
        header->initialize(StorageType::CONTINUOUS);
        // memset bitmap to initialize status
        header->reset();

        trace("[MMAP_MM] returning ", reinterpret_cast<void*>(aligned_ptr), " as new chunk");
        return reinterpret_cast<void*>(aligned_ptr);
    }

    // Just allocate with StorageType padded in front
    // size must be larger than 128MB
    // TODO: it must be aligned to work AT ALL
    void* allocateLarge(size_t size)
    {
        trace("Called for large object with size ", std::hex, size);
        //assert(size > ALLOCATION_SIZE);
        size += sizeof(ObjectInfo);
        size_t alloc_size = (size % LINUX_PAGE_SIZE == 0) ? size : (size - (size % LINUX_PAGE_SIZE) + LINUX_PAGE_SIZE);
        //assert(alloc_size % LINUX_PAGE_SIZE == 0);
        char* ptr = reinterpret_cast<char*>(mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, 0, 0));
        ObjectInfo* type = reinterpret_cast<ObjectInfo*>(ptr);
        type->size = alloc_size;
        ptr += sizeof(ObjectInfo);

        return ptr;
    }

    void deallocateLarge(void* const ptr)
    {
        trace("Called for dealloc large object with ptr ", ptr);
        ObjectInfo* info = reinterpret_cast<ObjectInfo*>( reinterpret_cast<uint64_t>(ptr) - sizeof(ObjectInfo));
        //assert(info->size % LINUX_PAGE_SIZE == 0);
        munmap(info, info->size);
    }

    void* allocatePages(size_t size, void* chunk_location)
    {
        trace( "Page allocation request for size ", std::hex, size, " on location ", chunk_location);
        if (size < DB_PAGE_SIZE) {
            size = DB_PAGE_SIZE;
        }

        ChunkHeader* header = reinterpret_cast<ChunkHeader*>(reinterpret_cast<uint64_t>(chunk_location) - sizeof(ChunkHeader));
        void* ptr = header->findNextAllocatableSlot(size);
        trace( "header on ", header, " found next allocatable slot as ", ptr);
        if (ptr == nullptr) {
            return nullptr;
        }
        header->setAllocated(ptr, size);

        return ptr;
    }

    void* allocate(size_t size, void* chunk_location) 
    {
        if (size > ALLOCATION_SIZE) {
            warn("Allocating large object of size");
            return allocateLarge(size);
        }
        else {
            if (chunk_location != nullptr) {
                trace("Allocating pages with chunk ", chunk_location);
                return allocatePages(size, chunk_location);
            }
            else if (m_current_chunk == nullptr) {
#ifdef USE_FREEMAP
                void* ptr = freemap.getMemory(size);
                if (ptr != nullptr)
                    return ptr;
#endif

                trace("Using allocator specific chunk on ", m_current_chunk);
                m_current_chunk = allocateContinuous();
            }

            return allocatePages(size, m_current_chunk);
        }
    }

    void* allocate(size_t size) override
    {
        if (size > ALLOCATION_SIZE) {
            void* ptr = allocateLarge(size);
            if (ptr == nullptr)
                throw std::runtime_error("Out of memory");

            trace("Returning pointer on ", ptr, " for size ", size);
            return ptr;
        }
        else {
#ifdef USE_FREEMAP
            void* freemapptr = freemap.getMemory(size);
            if (freemapptr != nullptr)
                return freemapptr;
#endif

            if (m_current_chunk == nullptr)
                m_current_chunk = allocateContinuous();

            void* ptr = allocatePages(size, m_current_chunk);
            if (ptr == nullptr) {
                debug("Needing new chunk, allocating...");
                m_current_chunk = allocateContinuous();

                if (m_current_chunk == nullptr)
                    throw std::runtime_error("Out of memory");

                ptr = allocatePages(size, m_current_chunk);
            }
            trace("Returning pointer on ", ptr, " for size ", size);
            return ptr;
        }
    }

    void deallocate(void* const ptr) override
    {
        trace("[MMAP_MM] Deallocating ptr on ", ptr);

#ifdef USE_FREEMAP
        ObjectInfo* info = reinterpret_cast<ObjectInfo*>(ptr);
        // Check if we can just insert into freemap
        if (!freemap.hasTooManyEntries(info->size)) {
            freemap.insert(ptr, info->size);
            return;
        }
#endif
        // if we have too many entries, lets dealloc
        uint64_t chunk_ptr = reinterpret_cast<uint64_t>(ptr) & ~(ALLOCATION_SIZE - 1);
        ChunkHeader* header = reinterpret_cast<ChunkHeader*>( chunk_ptr - sizeof(ChunkHeader) );

        header->setDeallocated(ptr);
        if (header->m_info.status.m_totalRegions == 0) {
            if (reinterpret_cast<void*>(chunk_ptr) == m_current_chunk)
                m_current_chunk = nullptr;
            //munmap(header, sizeof(ChunkHeader) + ALLOCATION_SIZE);
            if (m_next_free_chunk == nullptr) {
                m_next_free_chunk = reinterpret_cast<void*>(chunk_ptr);

                ChunkHeader::getChunkHeader(m_next_free_chunk)->setPrev(nullptr);
                ChunkHeader::getChunkHeader(m_next_free_chunk)->setNext(nullptr);
            }
            else {
                ChunkHeader* header_iter = ChunkHeader::getChunkHeader(m_next_free_chunk);
                while (header_iter->getNext() != nullptr) {
                   header_iter = ChunkHeader::getChunkHeader(header_iter->getNext());
                }

                // append to list
                header_iter->setNext(m_next_free_chunk);
                ChunkHeader::getChunkHeader(m_next_free_chunk)->setPrev(ChunkHeader::getChunkPointer(header_iter));
                ChunkHeader::getChunkHeader(m_next_free_chunk)->setNext(nullptr);
            }
        }
    }

    void deallocateAll()
    {

    }

    void *allocate(abstract_memory_manager *const /*manager*/, size_t size) override
    {
        throw std::runtime_error("Not implemented");
         return allocate(size);
    }

    void deallocate(abstract_memory_manager *const /*manager*/, void *const ptr) override
    {
        throw std::runtime_error("Not implemented");
        deallocate(ptr);
    }

    void * reallocate(abstract_memory_manager * const /*manager*/, void * ptr, size_t size) override
    {
        throw std::runtime_error("Not implemented");
        return reallocate(ptr, size);
    }

    void * reallocate(void* ptr, size_t size) override
    {
        trace("Called for realloc on ", ptr, " and size ", std::hex, size);
        return reallocate_back(ptr, size);
    }

    void * reallocate_front(void* /*ptr*/, size_t /*size*/)
    {
        //TODO: do proper implementation

        return nullptr;
    }

    bool isReallocatable(void* ptr, size_t size)
    {
        ChunkHeader* header = ChunkHeader::getChunkHeader(ptr);

        return header->isReallocatable(ptr, size);
    }

    void * reallocate_back(void* ptr, size_t size)
    {
        debug("[MMAP_MM] Called reallocate_back");

        ObjectInfo* info = reinterpret_cast<ObjectInfo*>(ptr);
        ChunkHeader* header = ChunkHeader::getChunkHeader(ptr);
        if (info->getSize() == size)
            return ptr;

        if (info->getSize() > size) {
            debug("Size ", size, " is smaller than former size ", info->getSize());
            header->setDeallocatedEnd(ptr, info->getSize(), size);
            info->size = size;
            return ptr;
        }

        void* ret = allocate(size);
        debug("Reallocating ", size, " bytes for ", info->getSize(), " by allocating anew");
        if (ret == nullptr) {
            throw std::runtime_error("Allocation failed");
            return nullptr;
        }
        memcpy(ret, ptr, info->size > size ? size : info->size);
        deallocate(ptr);
        info = reinterpret_cast<ObjectInfo*>(ret);
        info->size = size;
        return &info[1];
    }

    void handle_error() override
    {

    }

private:
    explicit mmap_memory_manager() : 
        abstract_memory_manager{},
        m_current_chunk(nullptr), m_next_free_chunk(nullptr),
        m_initialized{(
                          (stdlib_malloc_ptr == nullptr) ||
                          (stdlib_malloc_ptr == nullptr) ||
                          (stdlib_malloc_ptr == nullptr)
                       ) ? init_mem_hooks() : true
        }
#ifdef USE_FREEMAP
        ,
        freemap()
#endif
        {
             debug("setting mmap mm active");
             debug("setting mmap mm active - out");
        }

    // used for direct allocation calls
    void* m_current_chunk;
    void* m_next_free_chunk;
    bool m_initialized;
    std::mutex m_Mutex;
#ifdef USE_FREEMAP
    ListMap<ALLOCATION_SIZE,DB_PAGE_SIZE> freemap;
#endif
};

} //namespace morphstore

#endif //MORPHSTORE_CORE_MEMORY_MANAGEMENT_MMAP_MM_H
