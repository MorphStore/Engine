/**
 * @file group_simple.h
 * @brief The compression format/algorithm SIMD-Group-Simple.
 * 
 * The original SIMD-Group-Simple algorithm was proposed in the following paper:
 * 
 * Wayne Xin Zhao, Xudong Zhang, Daniel Lemire, Dongdong Shan, Jian-Yun Nie,
 * Hongfei Yan, Ji-Rong Wen: A General SIMD-Based Approach to Accelerating
 * Compression Algorithms. ACM Trans. Inf. Syst. 33(3): 15:1-15:28 (2015)
 * 
 * Our implementation differs in some points:
 * - It supports 64-bit or 32-bit data elements, depending on the base type of
 *   the specified processing style of the Template Vector Library (TVL).
 * - We do not use a large array, but a ring buffer for the pseudo group max
 *   values.
 * - Instead of one contiguous selectors area and one contiguous data area, a
 *   compressed buffer might contain multiple pairs of these areas. Each call
 *   of the compressing batch-level morph-operator creates a new pair. This is
 *   handy for the blockwise recompression used in MorphStore's on-the-fly
 *   de/re-compression operators. However, if all data in a column is
 *   compressed in a single call, then there is exactly one pair of selectors
 *   area and data area.
 */

#ifndef MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_H
#define MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_H

#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/group_simple_routines.h> // generated code
#include <core/morphing/morph.h>
#include <core/utils/basic_types.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_primitives.h>
#include <vector/vector_extension_structs.h>

#include <algorithm>
#include <limits>

#include <cstdint>

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    /**
     * @brief The format of SIMD-Group-Simple (with some deviations).
     * 
     * Our implementation of the algorithm produces data in the following
     * memory layout: The compressed data is a sequence of pages. Each page
     * consists of:
     * - a header
     *   - Containing some meta data. More precisely, that is the number of
     *     selectors as a m_count_sels_t, and the number of groups in the last,
     *     incomplete block as a m_count_remaining_groups_t.
     * - a selectors area
     *   - Containing the so-called selectors. The i-th selector indicates the
     *     compression mode used for the i-th data block. See
     *     `struct group_simple_tables` for a reference. The selectors are
     *     packed to 4 bit per selector in the horizontal layout.
     * - possibly a padding
     *   - to ensure alignment
     * - a data area
     *   - Containing the packed data elements in the vertical layout. Each
     *     compressed block has a size of one vector, i.e., the lower the bit
     *     width, the higher the number of data elements (variable block size
     *     in terms of logical data elements). See `struct group_simple_tables`
     *     for a reference. So there might be unsued bits in such a vector.
     * 
     * According to the original paper, there would be only one page per
     * column. This can be achieved by passing the entire data of an
     * uncompressed column to the compressing batch-level morph-operator at
     * once. However, when a compressed column is created through a blockwise
     * compression, i.e., when the compressing batch-level morph-operator is
     * called repeatedly block by block, then the compressed column will
     * contain multiple pages. The decompression (with processing) can handle
     * both cases.
     * 
     * @param t_GroupSizeLog The number of logical data elements per group. All
     * data elements of one group are viewed as their maximum value by the
     * routine determining the variable block sizes. Naturally, this is the
     * number of logical data elements per vector.
     * @param t_base_t The data type of a logical data element. This must be
     * known because it determines the available compression modes respectively
     * selectors. Currently, `uint64_t` and `uint32_t` are supported.
     * @param t_AlignmentByte The number of bytes to which the data area(s) is
     * (are) aligned. Naturally, this is the number of bytes per vector.
     */
    template<size_t t_GroupSizeLog, typename t_base_t, size_t t_AlignmentBytes>
    struct group_simple_f : public format {
        // This is somewhat pessimistic. uint32_t would usually suffice.
        using m_count_sels_t = uint64_t;
        // The number of remaining groups is always less than the bit width of
        // base_t. Thus, a single byte should always be sufficient.
        using m_count_remaining_groups_t = uint8_t;
        
        static size_t get_size_max_byte(size_t p_CountValues) {
            // In the worst case, every data element requires 64 bit
            // (respectively the bit width of the base type). In that case, we
            // need one selector byte per two groups of data elements
            // (remember that the selectors are packed to 4-bit).

            // @todo Don't hardcode 1024. This should be the minimal block size
            // (in terms of uncompressed data elements) which is ever used for
            // a blockwise compression using this format. For the recompression
            // in on-the-fly de/re-compression, that size is 2048 (by default),
            // but for the calibration benchmarks for cache-to-ram compression
            // it is just 1024. So we use 1024 here. However, we should find a
            // more general solution. In general, we need to know the
            // blocksize, because the meta data and padding exist for each
            // compressed block.
            const size_t maxComprBlockCount = round_up_div(p_CountValues, 1024);
            
            return
                    // meta data
                    maxComprBlockCount * (sizeof(m_count_sels_t) + sizeof(m_count_remaining_groups_t)) +
                    // selectors area
                    // (2 is due to packing two selectors into one byte.)
                    round_up_to_multiple(p_CountValues / t_GroupSizeLog / 2, 2) +
                    // alignment padding between selectors and data area
                    maxComprBlockCount * (t_AlignmentBytes - 1) +
                    // data area
                    sizeof(t_base_t) * p_CountValues;
        }
        
        static const size_t m_BlockSize = t_GroupSizeLog;
        
        MSV_CXX_ATTRIBUTE_INLINE
        static unsigned bitwidth_incomplete_block(size_t p_CountGroups) {
            // Since we have to produce exactly one compressed vector for the
            // incomplete block anyway, we can use the highest bit width
            // allowing us to pack all p_CountGroups values.
            return std::numeric_limits<t_base_t>::digits / p_CountGroups;
        }
    };
    
    // ************************************************************************
    // Lookup tables
    // ************************************************************************
    
    /**
     * @brief A container for the tables used by the Group-Simple algorithm.
     * 
     * They are not a part of `group_simple_f` because they only depend on the
     * base type, not on the other template parameters of `group_simple_f`.
     */
    template<typename t_base_t>
    struct group_simple_tables {
        static const uint8_t m_TableNum[];
        static const t_base_t m_TableMask[];
    };
    
    // ------------------------------------------------------------------------
    // For 64-bit base type
    // ------------------------------------------------------------------------
    // A compressed block can contain (gs is the group size, usually the number
    // of data elements per vector):
    // - gs x 64  1-bit values
    // - gs x 32  2-bit values
    // - ...
    // - gs x  1 64-bit value
    
    template<>
    const uint8_t group_simple_tables<uint64_t>::m_TableNum[] = {
        64, 32, 21, 16, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    };
    template<>
    const uint64_t group_simple_tables<uint64_t>::m_TableMask[] = {
        bitwidth_max<uint64_t>(1),
        bitwidth_max<uint64_t>(2),
        bitwidth_max<uint64_t>(3),
        bitwidth_max<uint64_t>(4),
        bitwidth_max<uint64_t>(5),
        bitwidth_max<uint64_t>(6),
        bitwidth_max<uint64_t>(7),
        bitwidth_max<uint64_t>(8),
        bitwidth_max<uint64_t>(9),
        bitwidth_max<uint64_t>(10),
        bitwidth_max<uint64_t>(12),
        bitwidth_max<uint64_t>(16),
        bitwidth_max<uint64_t>(21),
        bitwidth_max<uint64_t>(32),
        bitwidth_max<uint64_t>(64)
    };
    
    // ------------------------------------------------------------------------
    // For 32-bit base type
    // ------------------------------------------------------------------------
    // A compressed block can contain (gs is the group size, usually the number
    // of data elements per vector):
    // - gs x 32  1-bit values
    // - gs x 16  2-bit values
    // - ...
    // - gs x  1 32-bit value
    
    template<>
    const uint8_t group_simple_tables<uint32_t>::m_TableNum[] = {
        32, 16, 10, 8, 6, 5, 4, 3, 2, 1
    };
    template<>
    const uint32_t group_simple_tables<uint32_t>::m_TableMask[] = {
        bitwidth_max<uint32_t>(1),
        bitwidth_max<uint32_t>(2),
        bitwidth_max<uint32_t>(3),
        bitwidth_max<uint32_t>(4),
        bitwidth_max<uint32_t>(5),
        bitwidth_max<uint32_t>(6),
        bitwidth_max<uint32_t>(8),
        bitwidth_max<uint32_t>(10),
        bitwidth_max<uint32_t>(16),
        bitwidth_max<uint32_t>(32)
    };
    
    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension>
    class morph_batch_t<
            t_vector_extension,
            group_simple_f<
                    t_vector_extension::vector_helper_t::element_count::value,
                    typename t_vector_extension::base_t,
                    t_vector_extension::vector_helper_t::size_byte::value
            >,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using dst_f = group_simple_f<
                vector_element_count::value, base_t, vector_size_byte::value
        >;
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void compress_incomplete_block(
                typename dst_f::m_count_remaining_groups_t p_CountGroups,
                const base_t * & p_InBase,
                base_t * & p_OutBase
        ) {
            using namespace vectorlib;
            
            const unsigned bw = dst_f::bitwidth_incomplete_block(p_CountGroups);
            vector_t comprBlock = load<
                    t_ve, iov::ALIGNED, vector_size_bit::value
            >(p_InBase);
            for(size_t k = 1; k < p_CountGroups; k++)
                comprBlock = bitwise_or<t_ve, vector_size_bit::value>(
                        comprBlock,
                        shift_left<t_ve, vector_base_t_granularity::value>::apply(
                                load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                                        p_InBase + k * vector_element_count::value
                                ),
                                k * bw
                        )
                );
            store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                    p_OutBase, comprBlock
            );
            p_InBase += vector_element_count::value * p_CountGroups;
            p_OutBase += vector_element_count::value;
        }
        
    public:
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            using namespace morphstore::group_simple_routines;
            
            typename dst_f::m_count_sels_t * const countSelsPtr =
                    reinterpret_cast<typename dst_f::m_count_sels_t *>(p_Out8);
            typename dst_f::m_count_remaining_groups_t * const countRemainingGroupsPtr =
                    reinterpret_cast<typename dst_f::m_count_remaining_groups_t *>(
                            p_Out8 + sizeof(typename dst_f::m_count_sels_t)
                    );
            uint8_t * selArea = p_Out8 +
                    sizeof(typename dst_f::m_count_sels_t) +
                    sizeof(typename dst_f::m_count_remaining_groups_t);
            const uint8_t * const initSelArea = selArea;
            
            const size_t rbMaxSize = std::numeric_limits<base_t>::digits;
            base_t groupMaxRb[rbMaxSize];
            size_t rbPos = 0;
            size_t rbSize = 0;
            
            // Step 1 and 2: Determine the compression mode/selector for all
            // blocks.
            bool even = true;
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            const base_t * const initInBase = inBase;
            const base_t * const endInBase = inBase + p_CountLog;
            while(inBase < endInBase || rbSize) {
                // Step 1: Refill the group max ring buffer.
                const size_t countRemainingGroups =
                        static_cast<size_t>(endInBase - inBase) /
                        vector_element_count::value;
                const size_t rbSizeToReach = std::min(
                        rbMaxSize, rbSize + countRemainingGroups
                );
                for(; rbSize < rbSizeToReach; rbSize++) {
                    const base_t pseudoGroupMax = hor<
                            t_ve, vector_base_t_granularity::value
                    >::apply(load<t_ve, iov::ALIGNED, vector_size_bit::value>(
                            inBase
                    ));
                    inBase += vector_element_count::value;
                    groupMaxRb[(rbPos + rbSize) % rbMaxSize] = pseudoGroupMax;
                }

                // Step 2: Determine the next selector.
                unsigned i;
                size_t n;
                size_t offset;
                // Note that the number of possible compression modes depends
                // on the width of the base type.
                const unsigned countAvailableComprModes =
                        sizeof(group_simple_tables<base_t>::m_TableMask) /
                        sizeof(base_t);
                for(i = 0; i < countAvailableComprModes; i++) {
                    n = group_simple_tables<base_t>::m_TableNum[i];
                    offset = 0;
                    const size_t maxPos = std::min(n, rbSize);
                    const base_t mask = group_simple_tables<base_t>::m_TableMask[i];
                    while(offset < maxPos && groupMaxRb[(rbPos + offset) % rbMaxSize] <= mask)
                        offset++;
                    if(offset == maxPos)
                        break;
                }
                if(offset == n) {
                    // Store the selector.
                    if(even)
                        *selArea = i;
                    else
                        *selArea++ |= (i << 4);
                    even = !even;

                    // Update the ring buffer.
                    rbPos = (rbPos + n) % rbMaxSize;
                    rbSize -= n;
                }
                else
                    break;
            }
            
            // Counts only the full selector bytes.
            const size_t selAreaSizeByte =
                    static_cast<size_t>(selArea - initSelArea);
            const typename dst_f::m_count_sels_t countSels =
                    2 * selAreaSizeByte + (even ? 0 : 1);
            *countSelsPtr = countSels;
            inBase = initInBase; // rewind
            base_t * dataArea = reinterpret_cast<base_t *>(create_aligned_ptr(
                    selArea + (even ? 0 : 1), vector_size_byte::value
            ));
            
            // Step 3: Compress all blocks.
            for(size_t i = 0; i < selAreaSizeByte; i++) {
                const uint8_t selByte = initSelArea[i];
                const uint8_t selLo =  selByte       & 0b1111;
                const uint8_t selHi = (selByte >> 4) & 0b1111;
                compress_complete_block<t_ve>(selLo, inBase, dataArea);
                compress_complete_block<t_ve>(selHi, inBase, dataArea);
            }
            if(!even) {
                const uint8_t selLo = initSelArea[selAreaSizeByte] & 0b1111;
                compress_complete_block<t_ve>(selLo, inBase, dataArea);
            }
            const typename dst_f::m_count_remaining_groups_t countRemainingGroups =
                (endInBase - inBase) / vector_element_count::value;
            *countRemainingGroupsPtr = countRemainingGroups;
            if(countRemainingGroups)
                compress_incomplete_block(
                        countRemainingGroups, inBase, dataArea
                );
            
            p_In8 = reinterpret_cast<const uint8_t *>(inBase);
            p_Out8 = reinterpret_cast<uint8_t *>(dataArea);
        }
    };
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<class t_vector_extension>
    class morph_batch_t<
            t_vector_extension,
            uncompr_f,
            group_simple_f<
                    t_vector_extension::vector_helper_t::element_count::value,
                    typename t_vector_extension::base_t,
                    t_vector_extension::vector_helper_t::size_byte::value
            >
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = group_simple_f<
                vector_element_count::value, base_t, vector_size_byte::value
        >;
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void decompress_incomplete_block(
                typename src_f::m_count_remaining_groups_t p_CountGroups,
                const base_t * & p_InBase,
                base_t * & p_OutBase
        ) {
            using namespace vectorlib;
            
            const unsigned bw = src_f::bitwidth_incomplete_block(p_CountGroups);
            const vector_t mask = set1<t_ve, vector_base_t_granularity::value>(
                    bitwidth_max<base_t>(bw)
            );
            const vector_t comprBlock = load<
                    t_ve, iov::ALIGNED, vector_size_bit::value
            >(p_InBase);
            for(size_t k = 0; k < p_CountGroups; k++)
                store<t_ve, iov::ALIGNED, vector_size_bit::value>(
                        p_OutBase + k * vector_element_count::value,
                        bitwise_and<t_ve, vector_size_bit::value>(
                                shift_right<
                                        t_ve, vector_base_t_granularity::value
                                >::apply(
                                        comprBlock, k * bw
                                ),
                                mask
                        )
                );
            
            p_InBase += vector_element_count::value;
            p_OutBase += vector_element_count::value * p_CountGroups;
        }
        
    public:
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            using namespace morphstore::group_simple_routines;
            
            base_t * outBase = reinterpret_cast<base_t *>(p_Out8);
            const base_t * const initOutBase = outBase;
            
            while(static_cast<size_t>(outBase - initOutBase) < p_CountLog) {
                const typename src_f::m_count_sels_t countSels =
                        *reinterpret_cast<const typename src_f::m_count_sels_t *>(p_In8);
                p_In8 += sizeof(typename src_f::m_count_sels_t);
                const typename src_f::m_count_remaining_groups_t countRemainingGroups =
                        *reinterpret_cast<const typename src_f::m_count_remaining_groups_t *>(p_In8);
                p_In8 += sizeof(typename src_f::m_count_remaining_groups_t);

                const uint8_t * const selArea = p_In8;
                const base_t * dataArea = reinterpret_cast<const base_t *>(
                        create_aligned_ptr(
                                p_In8 + round_up_div(countSels, 2),
                                vector_size_byte::value
                        )
                );

                const size_t countFullSelBytes = countSels / 2;
                for(size_t i = 0; i < countFullSelBytes; i++) {
                    const uint8_t selByte = selArea[i];
                    const uint8_t selLo =  selByte       & 0b1111;
                    const uint8_t selHi = (selByte >> 4) & 0b1111;
                    decompress_complete_block<t_ve>(selLo, dataArea, outBase);
                    decompress_complete_block<t_ve>(selHi, dataArea, outBase);
                }
                if(countSels % 2) {
                    const uint8_t sel = selArea[countFullSelBytes] & 0b1111;
                    decompress_complete_block<t_ve>(sel, dataArea, outBase);
                }
                if(countRemainingGroups)
                    decompress_incomplete_block(
                            countRemainingGroups, dataArea, outBase
                    );

                p_In8 = reinterpret_cast<const uint8_t *>(dataArea);
                p_Out8 = reinterpret_cast<uint8_t *>(outBase);
            }
        }
    };
    
    // ************************************************************************
    // Interfaces for accessing compressed data
    // ************************************************************************

    // ------------------------------------------------------------------------
    // Sequential read
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    class decompress_and_process_batch<
            t_vector_extension,
            group_simple_f<
                    t_vector_extension::vector_helper_t::element_count::value,
                    typename t_vector_extension::base_t,
                    t_vector_extension::vector_helper_t::size_byte::value
            >,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = group_simple_f<
                vector_element_count::value, base_t, vector_size_byte::value
        >;
        
        MSV_CXX_ATTRIBUTE_FORCE_INLINE
        static void decompress_and_process_incomplete_block(
                typename in_f::m_count_remaining_groups_t p_CountGroups,
                const base_t * & p_InBase,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            using namespace vectorlib;

            const unsigned bw = in_f::bitwidth_incomplete_block(p_CountGroups);
            const vector_t mask = set1<t_ve, vector_base_t_granularity::value>(
                    bitwidth_max<base_t>(bw)
            );
            const vector_t comprBlock =
                    load<t_ve, iov::ALIGNED, vector_size_bit::value>(p_InBase);
            for(size_t k = 0; k < p_CountGroups; k++)
                t_op_vector<t_ve, t_extra_args ...>::apply(
                        bitwise_and<t_ve, vector_size_bit::value>(
                                shift_right<
                                        t_ve, vector_base_t_granularity::value
                                >::apply(
                                        comprBlock, k * bw
                                ),
                                mask
                        ),
                        p_State
                );
            
            p_InBase += vector_element_count::value;
        }
        
    public:
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            using namespace vectorlib;
            using namespace morphstore::group_simple_routines;
            
            // We implement the mechanism for detecting when we are done a
            // little different then in the decompression, since here there is
            // no output pointer we could refer to. But this incurs extra
            // additions on countProcessedGroups.
            size_t countProcessedGroups = 0;
            const size_t countGroupsToProcess = p_CountInLog / in_f::m_BlockSize;
            
            while(countProcessedGroups < countGroupsToProcess) {
                const typename in_f::m_count_sels_t countSels =
                        *reinterpret_cast<const typename in_f::m_count_sels_t *>(p_In8);
                p_In8 += sizeof(typename in_f::m_count_sels_t);
                const typename in_f::m_count_remaining_groups_t countRemainingGroups =
                        *reinterpret_cast<const typename in_f::m_count_remaining_groups_t *>(p_In8);
                p_In8 += sizeof(typename in_f::m_count_remaining_groups_t);

                const uint8_t * const selArea = p_In8;
                const base_t * dataArea = reinterpret_cast<const base_t *>(
                        create_aligned_ptr(
                                p_In8 + round_up_div(countSels, 2),
                                vector_size_byte::value
                        )
                );

                const size_t countFullSelBytes = countSels / 2;
                for(size_t i = 0; i < countFullSelBytes; i++) {
                    const uint8_t selByte = selArea[i];
                    const uint8_t selLo =  selByte       & 0b1111;
                    const uint8_t selHi = (selByte >> 4) & 0b1111;
                    decompress_and_process_complete_block<
                            t_ve, t_op_vector, t_extra_args ...
                    >(selLo, dataArea, p_State);
                    decompress_and_process_complete_block<
                            t_ve, t_op_vector, t_extra_args ...
                    >(selHi, dataArea, p_State);
                    countProcessedGroups +=
                            group_simple_tables<base_t>::m_TableNum[selLo];
                    countProcessedGroups +=
                            group_simple_tables<base_t>::m_TableNum[selHi];
                }
                if(countSels % 2) {
                    const uint8_t sel = selArea[countFullSelBytes] & 0b1111;
                    decompress_and_process_complete_block<
                            t_ve, t_op_vector, t_extra_args ...
                    >(sel, dataArea, p_State);
                    countProcessedGroups +=
                            group_simple_tables<base_t>::m_TableNum[sel];
                }
                if(countRemainingGroups) {
                    decompress_and_process_incomplete_block(
                            countRemainingGroups, dataArea, p_State
                    );
                    countProcessedGroups += countRemainingGroups;
                }

                p_In8 = reinterpret_cast<const uint8_t *>(dataArea);
            }
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_GROUP_SIMPLE_H
