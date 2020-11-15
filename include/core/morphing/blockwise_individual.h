/**
 * @file blockwise_individual.h
 * @brief A meta-format for using an individual format for each block of a
 * column.
 */

#ifndef MORPHSTORE_CORE_MORPHING_BLOCKWISE_INDIVIDUAL_H
#define MORPHSTORE_CORE_MORPHING_BLOCKWISE_INDIVIDUAL_H

#include <core/format_selection/format_selector_commons.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/format.h>
#include <core/morphing/group_simple.h>
#include <core/morphing/morph.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/utils/basic_types.h>
#include <vector/vector_extension_structs.h>

#include <algorithm>
#include <type_traits>

#include <cstdint>

/**
 * If this macro is defined, then an exception is thrown when a format selector
 * returns an unsupported format (compression) or when the meta data of a
 * compressed block indicates an unsupported format (decompression,
 * decompression and processing). Switching this feature on might have a
 * negative impact on performance.
 */
#undef BWI_CHECK_FORMAT_CODE
#ifdef BWI_CHECK_FORMAT_CODE
#include <stdexcept>
#endif

namespace morphstore {
    
    // ************************************************************************
    // Format
    // ************************************************************************
    
    /**
     * @brief A meta-format for using an individual format for each block of a
     * column.
     * 
     * The column is subdivided into blocks of `t_BlockSizeLog` logical data
     * elements each. For each block, an individual format is chosen during the
     * compression using the format selector `t_format_selector`. Note that
     * these inner formats are black boxes to `blockwise_individual_f`.
     * 
     * The compressed representation is structured as follows: It is a sequence
     * of compressed blocks. Each compressed block consists of the following
     * (in this order):
     * - a single byte indicating the format it is represented in (see
     *   `format_code`)
     * - some padding, i.e. unused bytes, to achieve memory alignment for SIMD
     *   load/store
     * - the compressed data in the format chosen for this block
     * - after that, there is no extra padding, because it is okay for the meta
     *   data of the next block to be unaligned
     * 
     * Note that providing the processing style (vector extension) as a
     * template parameter of the format is necessary here, because it
     * simplifies things a lot, especially when deriving the default template
     * parameters of the inner formats. Since these inner formats are black
     * boxes, this would be hard to accomplish otherwise. Nevertheless, in
     * general, the processing style should *not* be known to a format, since
     * it only concerns the processing.
     */
    template<
            size_t t_BlockSizeLog,
            template<class /*t_vector_extension*/> class t_format_selector,
            class t_vector_extension
    >
    struct blockwise_individual_f : public format {
        using t_ve = t_vector_extension;
        
        static_assert(
                std::is_base_of<
                        format_selector<t_vector_extension>,
                        t_format_selector<t_vector_extension>
                >::value,
                "blockwise_individual_f: the template parameter "
                "t_format_selector must be a subclass of format_selector"
        );
        
        #define BWI_STATIC_ASSERT_BLOCKSIZE(format) \
            static_assert( \
                    t_BlockSizeLog % format::m_BlockSize == 0, \
                    "blockwise_individual_f: the logical block size of blockwise_individual_f must be a " \
                    "multiple of the logical block size of each format supported for an individual block" \
            );
        BWI_STATIC_ASSERT_BLOCKSIZE(uncompr_f)
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_STATIC_VBP_F(t_ve, 64))
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_DYNAMIC_VBP_F(t_ve))
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_GROUP_SIMPLE_F(t_ve))
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_DELTA_DYNAMIC_VBP_F(t_ve))
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_DELTA_GROUP_SIMPLE_F(t_ve))
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_FOR_DYNAMIC_VBP_F(t_ve))
        BWI_STATIC_ASSERT_BLOCKSIZE(DEFAULT_FOR_GROUP_SIMPLE_F(t_ve))
        #undef BWI_STATIC_ASSERT_BLOCKSIZE
        
        static size_t get_size_max_byte(size_t p_CountValues) {
            // Determine the maximum size a block of t_BlockSizeLog logical
            // data elements could have in any of the supported inner formats.
            const size_t sizesMaxByteSingleBlock[] = {
                uncompr_f::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_STATIC_VBP_F        (t_ve, 64)::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_DYNAMIC_VBP_F       (t_ve)::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_GROUP_SIMPLE_F      (t_ve)::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_DELTA_DYNAMIC_VBP_F (t_ve)::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_DELTA_GROUP_SIMPLE_F(t_ve)::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_FOR_DYNAMIC_VBP_F   (t_ve)::get_size_max_byte(t_BlockSizeLog),
                DEFAULT_FOR_GROUP_SIMPLE_F  (t_ve)::get_size_max_byte(t_BlockSizeLog),
            };
            const size_t countSupportedFormats =
                    sizeof(sizesMaxByteSingleBlock) / sizeof(size_t);
            const size_t sizeMaxByteSingleBlock = *std::max_element(
                    sizesMaxByteSingleBlock,
                    sizesMaxByteSingleBlock + countSupportedFormats
            );
            
            return 
                    // number of blocks
                    p_CountValues / t_BlockSizeLog * (
                            // maximum size of the payload of block
                            sizeMaxByteSingleBlock +
                            // size of the metadata and padding of
                            // blockwise_individual_f
                            t_ve::vector_helper_t::size_byte::value
                    );
        }
        
        static const size_t m_BlockSize = t_BlockSizeLog;
    };
    
    // ************************************************************************
    // Morph-operators (batch-level)
    // ************************************************************************
    
    // The morph-operators need one case per possible/supported inner format of
    // an individual block. Restrictions to the set of available formats are
    // the responsibility of the format selector.
    
    // ------------------------------------------------------------------------
    // Compression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSizeLog,
            template<class /*t_vector_extension*/> class t_format_selector
    >
    class morph_batch_t<
            t_vector_extension,
            blockwise_individual_f<
                    t_BlockSizeLog, t_format_selector, t_vector_extension
            >,
            uncompr_f
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using dst_f = blockwise_individual_f<
                t_BlockSizeLog, t_format_selector, t_vector_extension
        >;
        
    public:
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            const base_t * inBase = reinterpret_cast<const base_t *>(p_In8);
            const base_t * const inBaseEnd = inBase + p_CountLog;
            
            while(inBase < inBaseEnd) {
                // Decide the format of this block.
                const format_code formatCode = 
                        t_format_selector<t_ve>::choose(inBase, t_BlockSizeLog);
                
                // Store the format code as the block's meta data.
                const uint8_t formatCodeInt = static_cast<uint8_t>(formatCode);
                *p_Out8++ = formatCodeInt;
                
                // Add the padding to achieve SIMD alignment.
                p_Out8 = create_aligned_ptr(p_Out8, vector_size_byte::value);
                
                // Delegate the actual compression to the chosen format.
                const uint8_t * tmpIn8 =
                        reinterpret_cast<const uint8_t *>(inBase);
                switch(formatCode) {
                    // Template for a single case.
                    #define BWI_CASE_COMPR(formatCode, format) \
                        case format_code::formatCode: \
                            morph_batch<t_ve, format, uncompr_f>( \
                                    tmpIn8, p_Out8, t_BlockSizeLog \
                            ); \
                            break;
                    // Handling of all non-static_vbp formats.
                    BWI_CASE_COMPR(uncompr, uncompr_f)
                    BWI_CASE_COMPR(defaultDynamicVbp, DEFAULT_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_COMPR(defaultGroupSimple, DEFAULT_GROUP_SIMPLE_F(t_ve))
                    BWI_CASE_COMPR(defaultDeltaDynamicVbp, DEFAULT_DELTA_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_COMPR(defaultDeltaGroupSimple, DEFAULT_DELTA_GROUP_SIMPLE_F(t_ve))
                    BWI_CASE_COMPR(defaultForDynamicVbp, DEFAULT_FOR_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_COMPR(defaultForGroupSimple, DEFAULT_FOR_GROUP_SIMPLE_F(t_ve))
                    // Special treatment for static_vbp.
                    // Generated with Python:
                    // for bw in range(1, 64 + 1):
                    //   print("BWI_CASE_COMPR(defaultStaticVbp_{}, DEFAULT_STATIC_VBP_F(t_ve, {}))".format(bw, bw))
                    BWI_CASE_COMPR(defaultStaticVbp_1, DEFAULT_STATIC_VBP_F(t_ve, 1))
                    BWI_CASE_COMPR(defaultStaticVbp_2, DEFAULT_STATIC_VBP_F(t_ve, 2))
                    BWI_CASE_COMPR(defaultStaticVbp_3, DEFAULT_STATIC_VBP_F(t_ve, 3))
                    BWI_CASE_COMPR(defaultStaticVbp_4, DEFAULT_STATIC_VBP_F(t_ve, 4))
                    BWI_CASE_COMPR(defaultStaticVbp_5, DEFAULT_STATIC_VBP_F(t_ve, 5))
                    BWI_CASE_COMPR(defaultStaticVbp_6, DEFAULT_STATIC_VBP_F(t_ve, 6))
                    BWI_CASE_COMPR(defaultStaticVbp_7, DEFAULT_STATIC_VBP_F(t_ve, 7))
                    BWI_CASE_COMPR(defaultStaticVbp_8, DEFAULT_STATIC_VBP_F(t_ve, 8))
                    BWI_CASE_COMPR(defaultStaticVbp_9, DEFAULT_STATIC_VBP_F(t_ve, 9))
                    BWI_CASE_COMPR(defaultStaticVbp_10, DEFAULT_STATIC_VBP_F(t_ve, 10))
                    BWI_CASE_COMPR(defaultStaticVbp_11, DEFAULT_STATIC_VBP_F(t_ve, 11))
                    BWI_CASE_COMPR(defaultStaticVbp_12, DEFAULT_STATIC_VBP_F(t_ve, 12))
                    BWI_CASE_COMPR(defaultStaticVbp_13, DEFAULT_STATIC_VBP_F(t_ve, 13))
                    BWI_CASE_COMPR(defaultStaticVbp_14, DEFAULT_STATIC_VBP_F(t_ve, 14))
                    BWI_CASE_COMPR(defaultStaticVbp_15, DEFAULT_STATIC_VBP_F(t_ve, 15))
                    BWI_CASE_COMPR(defaultStaticVbp_16, DEFAULT_STATIC_VBP_F(t_ve, 16))
                    BWI_CASE_COMPR(defaultStaticVbp_17, DEFAULT_STATIC_VBP_F(t_ve, 17))
                    BWI_CASE_COMPR(defaultStaticVbp_18, DEFAULT_STATIC_VBP_F(t_ve, 18))
                    BWI_CASE_COMPR(defaultStaticVbp_19, DEFAULT_STATIC_VBP_F(t_ve, 19))
                    BWI_CASE_COMPR(defaultStaticVbp_20, DEFAULT_STATIC_VBP_F(t_ve, 20))
                    BWI_CASE_COMPR(defaultStaticVbp_21, DEFAULT_STATIC_VBP_F(t_ve, 21))
                    BWI_CASE_COMPR(defaultStaticVbp_22, DEFAULT_STATIC_VBP_F(t_ve, 22))
                    BWI_CASE_COMPR(defaultStaticVbp_23, DEFAULT_STATIC_VBP_F(t_ve, 23))
                    BWI_CASE_COMPR(defaultStaticVbp_24, DEFAULT_STATIC_VBP_F(t_ve, 24))
                    BWI_CASE_COMPR(defaultStaticVbp_25, DEFAULT_STATIC_VBP_F(t_ve, 25))
                    BWI_CASE_COMPR(defaultStaticVbp_26, DEFAULT_STATIC_VBP_F(t_ve, 26))
                    BWI_CASE_COMPR(defaultStaticVbp_27, DEFAULT_STATIC_VBP_F(t_ve, 27))
                    BWI_CASE_COMPR(defaultStaticVbp_28, DEFAULT_STATIC_VBP_F(t_ve, 28))
                    BWI_CASE_COMPR(defaultStaticVbp_29, DEFAULT_STATIC_VBP_F(t_ve, 29))
                    BWI_CASE_COMPR(defaultStaticVbp_30, DEFAULT_STATIC_VBP_F(t_ve, 30))
                    BWI_CASE_COMPR(defaultStaticVbp_31, DEFAULT_STATIC_VBP_F(t_ve, 31))
                    BWI_CASE_COMPR(defaultStaticVbp_32, DEFAULT_STATIC_VBP_F(t_ve, 32))
                    BWI_CASE_COMPR(defaultStaticVbp_33, DEFAULT_STATIC_VBP_F(t_ve, 33))
                    BWI_CASE_COMPR(defaultStaticVbp_34, DEFAULT_STATIC_VBP_F(t_ve, 34))
                    BWI_CASE_COMPR(defaultStaticVbp_35, DEFAULT_STATIC_VBP_F(t_ve, 35))
                    BWI_CASE_COMPR(defaultStaticVbp_36, DEFAULT_STATIC_VBP_F(t_ve, 36))
                    BWI_CASE_COMPR(defaultStaticVbp_37, DEFAULT_STATIC_VBP_F(t_ve, 37))
                    BWI_CASE_COMPR(defaultStaticVbp_38, DEFAULT_STATIC_VBP_F(t_ve, 38))
                    BWI_CASE_COMPR(defaultStaticVbp_39, DEFAULT_STATIC_VBP_F(t_ve, 39))
                    BWI_CASE_COMPR(defaultStaticVbp_40, DEFAULT_STATIC_VBP_F(t_ve, 40))
                    BWI_CASE_COMPR(defaultStaticVbp_41, DEFAULT_STATIC_VBP_F(t_ve, 41))
                    BWI_CASE_COMPR(defaultStaticVbp_42, DEFAULT_STATIC_VBP_F(t_ve, 42))
                    BWI_CASE_COMPR(defaultStaticVbp_43, DEFAULT_STATIC_VBP_F(t_ve, 43))
                    BWI_CASE_COMPR(defaultStaticVbp_44, DEFAULT_STATIC_VBP_F(t_ve, 44))
                    BWI_CASE_COMPR(defaultStaticVbp_45, DEFAULT_STATIC_VBP_F(t_ve, 45))
                    BWI_CASE_COMPR(defaultStaticVbp_46, DEFAULT_STATIC_VBP_F(t_ve, 46))
                    BWI_CASE_COMPR(defaultStaticVbp_47, DEFAULT_STATIC_VBP_F(t_ve, 47))
                    BWI_CASE_COMPR(defaultStaticVbp_48, DEFAULT_STATIC_VBP_F(t_ve, 48))
                    BWI_CASE_COMPR(defaultStaticVbp_49, DEFAULT_STATIC_VBP_F(t_ve, 49))
                    BWI_CASE_COMPR(defaultStaticVbp_50, DEFAULT_STATIC_VBP_F(t_ve, 50))
                    BWI_CASE_COMPR(defaultStaticVbp_51, DEFAULT_STATIC_VBP_F(t_ve, 51))
                    BWI_CASE_COMPR(defaultStaticVbp_52, DEFAULT_STATIC_VBP_F(t_ve, 52))
                    BWI_CASE_COMPR(defaultStaticVbp_53, DEFAULT_STATIC_VBP_F(t_ve, 53))
                    BWI_CASE_COMPR(defaultStaticVbp_54, DEFAULT_STATIC_VBP_F(t_ve, 54))
                    BWI_CASE_COMPR(defaultStaticVbp_55, DEFAULT_STATIC_VBP_F(t_ve, 55))
                    BWI_CASE_COMPR(defaultStaticVbp_56, DEFAULT_STATIC_VBP_F(t_ve, 56))
                    BWI_CASE_COMPR(defaultStaticVbp_57, DEFAULT_STATIC_VBP_F(t_ve, 57))
                    BWI_CASE_COMPR(defaultStaticVbp_58, DEFAULT_STATIC_VBP_F(t_ve, 58))
                    BWI_CASE_COMPR(defaultStaticVbp_59, DEFAULT_STATIC_VBP_F(t_ve, 59))
                    BWI_CASE_COMPR(defaultStaticVbp_60, DEFAULT_STATIC_VBP_F(t_ve, 60))
                    BWI_CASE_COMPR(defaultStaticVbp_61, DEFAULT_STATIC_VBP_F(t_ve, 61))
                    BWI_CASE_COMPR(defaultStaticVbp_62, DEFAULT_STATIC_VBP_F(t_ve, 62))
                    BWI_CASE_COMPR(defaultStaticVbp_63, DEFAULT_STATIC_VBP_F(t_ve, 63))
                    BWI_CASE_COMPR(defaultStaticVbp_64, DEFAULT_STATIC_VBP_F(t_ve, 64))
                    // Undefine the case template.
                    #undef BWI_CASE_COMPR
#ifdef BWI_CHECK_FORMAT_CODE
                    default:
                        throw std::runtime_error(
                                "unsupported format chosen in "
                                "blockwise_individual_f compression: " +
                                std::to_string(formatCodeInt)
                        );
#endif
                }
                inBase += t_BlockSizeLog;
            }
            
            p_In8 = reinterpret_cast<const uint8_t *>(inBase);
        }
    };
    
    
    // ------------------------------------------------------------------------
    // Decompression
    // ------------------------------------------------------------------------
    
    template<
            class t_vector_extension,
            size_t t_BlockSizeLog,
            template<class /*t_vector_extension*/> class t_format_selector
    >
    class morph_batch_t<
            t_vector_extension,
            uncompr_f,
            blockwise_individual_f<
                    t_BlockSizeLog, t_format_selector, t_vector_extension
            >
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using src_f = blockwise_individual_f<
                t_BlockSizeLog, t_format_selector, t_vector_extension
        >;
        
    public:
        static void apply(
                const uint8_t * & p_In8, uint8_t * & p_Out8, size_t p_CountLog
        ) {
            using namespace vectorlib;
            
            size_t countDecompr = 0;
            
            while(countDecompr < p_CountLog) {
                // Find out which format is used for the current block.
                format_code formatCode = static_cast<format_code>(*p_In8++);
                
                // Skip the padding to find the compressed payload.
                p_In8 = create_aligned_ptr(p_In8, vector_size_byte::value);
                
                // Delegate the decompression to the used format.
                switch(formatCode) {
                    // Template for a single case.
                    #define BWI_CASE_DECOMPR(formatCode, format) \
                        case format_code::formatCode: \
                            morph_batch<t_ve, uncompr_f, format>( \
                                    p_In8, p_Out8, t_BlockSizeLog \
                            ); \
                            break;
                    // Handling of all non-static_vbp formats.
                    BWI_CASE_DECOMPR(uncompr, uncompr_f)
                    BWI_CASE_DECOMPR(defaultDynamicVbp, DEFAULT_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_DECOMPR(defaultGroupSimple, DEFAULT_GROUP_SIMPLE_F(t_ve))
                    BWI_CASE_DECOMPR(defaultDeltaDynamicVbp, DEFAULT_DELTA_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_DECOMPR(defaultDeltaGroupSimple, DEFAULT_DELTA_GROUP_SIMPLE_F(t_ve))
                    BWI_CASE_DECOMPR(defaultForDynamicVbp, DEFAULT_FOR_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_DECOMPR(defaultForGroupSimple, DEFAULT_FOR_GROUP_SIMPLE_F(t_ve))
                    // Special treatment for static_vbp.
                    // Generated with Python:
                    // for bw in range(1, 64 + 1):
                    //   print("BWI_CASE_DECOMPR(defaultStaticVbp_{}, DEFAULT_STATIC_VBP_F(t_ve, {}))".format(bw, bw))
                    BWI_CASE_DECOMPR(defaultStaticVbp_1, DEFAULT_STATIC_VBP_F(t_ve, 1))
                    BWI_CASE_DECOMPR(defaultStaticVbp_2, DEFAULT_STATIC_VBP_F(t_ve, 2))
                    BWI_CASE_DECOMPR(defaultStaticVbp_3, DEFAULT_STATIC_VBP_F(t_ve, 3))
                    BWI_CASE_DECOMPR(defaultStaticVbp_4, DEFAULT_STATIC_VBP_F(t_ve, 4))
                    BWI_CASE_DECOMPR(defaultStaticVbp_5, DEFAULT_STATIC_VBP_F(t_ve, 5))
                    BWI_CASE_DECOMPR(defaultStaticVbp_6, DEFAULT_STATIC_VBP_F(t_ve, 6))
                    BWI_CASE_DECOMPR(defaultStaticVbp_7, DEFAULT_STATIC_VBP_F(t_ve, 7))
                    BWI_CASE_DECOMPR(defaultStaticVbp_8, DEFAULT_STATIC_VBP_F(t_ve, 8))
                    BWI_CASE_DECOMPR(defaultStaticVbp_9, DEFAULT_STATIC_VBP_F(t_ve, 9))
                    BWI_CASE_DECOMPR(defaultStaticVbp_10, DEFAULT_STATIC_VBP_F(t_ve, 10))
                    BWI_CASE_DECOMPR(defaultStaticVbp_11, DEFAULT_STATIC_VBP_F(t_ve, 11))
                    BWI_CASE_DECOMPR(defaultStaticVbp_12, DEFAULT_STATIC_VBP_F(t_ve, 12))
                    BWI_CASE_DECOMPR(defaultStaticVbp_13, DEFAULT_STATIC_VBP_F(t_ve, 13))
                    BWI_CASE_DECOMPR(defaultStaticVbp_14, DEFAULT_STATIC_VBP_F(t_ve, 14))
                    BWI_CASE_DECOMPR(defaultStaticVbp_15, DEFAULT_STATIC_VBP_F(t_ve, 15))
                    BWI_CASE_DECOMPR(defaultStaticVbp_16, DEFAULT_STATIC_VBP_F(t_ve, 16))
                    BWI_CASE_DECOMPR(defaultStaticVbp_17, DEFAULT_STATIC_VBP_F(t_ve, 17))
                    BWI_CASE_DECOMPR(defaultStaticVbp_18, DEFAULT_STATIC_VBP_F(t_ve, 18))
                    BWI_CASE_DECOMPR(defaultStaticVbp_19, DEFAULT_STATIC_VBP_F(t_ve, 19))
                    BWI_CASE_DECOMPR(defaultStaticVbp_20, DEFAULT_STATIC_VBP_F(t_ve, 20))
                    BWI_CASE_DECOMPR(defaultStaticVbp_21, DEFAULT_STATIC_VBP_F(t_ve, 21))
                    BWI_CASE_DECOMPR(defaultStaticVbp_22, DEFAULT_STATIC_VBP_F(t_ve, 22))
                    BWI_CASE_DECOMPR(defaultStaticVbp_23, DEFAULT_STATIC_VBP_F(t_ve, 23))
                    BWI_CASE_DECOMPR(defaultStaticVbp_24, DEFAULT_STATIC_VBP_F(t_ve, 24))
                    BWI_CASE_DECOMPR(defaultStaticVbp_25, DEFAULT_STATIC_VBP_F(t_ve, 25))
                    BWI_CASE_DECOMPR(defaultStaticVbp_26, DEFAULT_STATIC_VBP_F(t_ve, 26))
                    BWI_CASE_DECOMPR(defaultStaticVbp_27, DEFAULT_STATIC_VBP_F(t_ve, 27))
                    BWI_CASE_DECOMPR(defaultStaticVbp_28, DEFAULT_STATIC_VBP_F(t_ve, 28))
                    BWI_CASE_DECOMPR(defaultStaticVbp_29, DEFAULT_STATIC_VBP_F(t_ve, 29))
                    BWI_CASE_DECOMPR(defaultStaticVbp_30, DEFAULT_STATIC_VBP_F(t_ve, 30))
                    BWI_CASE_DECOMPR(defaultStaticVbp_31, DEFAULT_STATIC_VBP_F(t_ve, 31))
                    BWI_CASE_DECOMPR(defaultStaticVbp_32, DEFAULT_STATIC_VBP_F(t_ve, 32))
                    BWI_CASE_DECOMPR(defaultStaticVbp_33, DEFAULT_STATIC_VBP_F(t_ve, 33))
                    BWI_CASE_DECOMPR(defaultStaticVbp_34, DEFAULT_STATIC_VBP_F(t_ve, 34))
                    BWI_CASE_DECOMPR(defaultStaticVbp_35, DEFAULT_STATIC_VBP_F(t_ve, 35))
                    BWI_CASE_DECOMPR(defaultStaticVbp_36, DEFAULT_STATIC_VBP_F(t_ve, 36))
                    BWI_CASE_DECOMPR(defaultStaticVbp_37, DEFAULT_STATIC_VBP_F(t_ve, 37))
                    BWI_CASE_DECOMPR(defaultStaticVbp_38, DEFAULT_STATIC_VBP_F(t_ve, 38))
                    BWI_CASE_DECOMPR(defaultStaticVbp_39, DEFAULT_STATIC_VBP_F(t_ve, 39))
                    BWI_CASE_DECOMPR(defaultStaticVbp_40, DEFAULT_STATIC_VBP_F(t_ve, 40))
                    BWI_CASE_DECOMPR(defaultStaticVbp_41, DEFAULT_STATIC_VBP_F(t_ve, 41))
                    BWI_CASE_DECOMPR(defaultStaticVbp_42, DEFAULT_STATIC_VBP_F(t_ve, 42))
                    BWI_CASE_DECOMPR(defaultStaticVbp_43, DEFAULT_STATIC_VBP_F(t_ve, 43))
                    BWI_CASE_DECOMPR(defaultStaticVbp_44, DEFAULT_STATIC_VBP_F(t_ve, 44))
                    BWI_CASE_DECOMPR(defaultStaticVbp_45, DEFAULT_STATIC_VBP_F(t_ve, 45))
                    BWI_CASE_DECOMPR(defaultStaticVbp_46, DEFAULT_STATIC_VBP_F(t_ve, 46))
                    BWI_CASE_DECOMPR(defaultStaticVbp_47, DEFAULT_STATIC_VBP_F(t_ve, 47))
                    BWI_CASE_DECOMPR(defaultStaticVbp_48, DEFAULT_STATIC_VBP_F(t_ve, 48))
                    BWI_CASE_DECOMPR(defaultStaticVbp_49, DEFAULT_STATIC_VBP_F(t_ve, 49))
                    BWI_CASE_DECOMPR(defaultStaticVbp_50, DEFAULT_STATIC_VBP_F(t_ve, 50))
                    BWI_CASE_DECOMPR(defaultStaticVbp_51, DEFAULT_STATIC_VBP_F(t_ve, 51))
                    BWI_CASE_DECOMPR(defaultStaticVbp_52, DEFAULT_STATIC_VBP_F(t_ve, 52))
                    BWI_CASE_DECOMPR(defaultStaticVbp_53, DEFAULT_STATIC_VBP_F(t_ve, 53))
                    BWI_CASE_DECOMPR(defaultStaticVbp_54, DEFAULT_STATIC_VBP_F(t_ve, 54))
                    BWI_CASE_DECOMPR(defaultStaticVbp_55, DEFAULT_STATIC_VBP_F(t_ve, 55))
                    BWI_CASE_DECOMPR(defaultStaticVbp_56, DEFAULT_STATIC_VBP_F(t_ve, 56))
                    BWI_CASE_DECOMPR(defaultStaticVbp_57, DEFAULT_STATIC_VBP_F(t_ve, 57))
                    BWI_CASE_DECOMPR(defaultStaticVbp_58, DEFAULT_STATIC_VBP_F(t_ve, 58))
                    BWI_CASE_DECOMPR(defaultStaticVbp_59, DEFAULT_STATIC_VBP_F(t_ve, 59))
                    BWI_CASE_DECOMPR(defaultStaticVbp_60, DEFAULT_STATIC_VBP_F(t_ve, 60))
                    BWI_CASE_DECOMPR(defaultStaticVbp_61, DEFAULT_STATIC_VBP_F(t_ve, 61))
                    BWI_CASE_DECOMPR(defaultStaticVbp_62, DEFAULT_STATIC_VBP_F(t_ve, 62))
                    BWI_CASE_DECOMPR(defaultStaticVbp_63, DEFAULT_STATIC_VBP_F(t_ve, 63))
                    BWI_CASE_DECOMPR(defaultStaticVbp_64, DEFAULT_STATIC_VBP_F(t_ve, 64))
                    // Undefine the case template.
                    #undef BWI_CASE_DECOMPR
#ifdef BWI_CHECK_FORMAT_CODE
                    default:
                        throw std::runtime_error(
                                "unsupported format detected in "
                                "blockwise_individual_f decompression: " +
                                std::to_string(static_cast<uint8_t>(formatCode))
                        );
#endif
                }
                countDecompr += t_BlockSizeLog;
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
            size_t t_BlockSizeLog,
            template<class /*t_vector_extension*/> class t_format_selector,
            template<class, class ...> class t_op_vector,
            class ... t_extra_args
    >
    class decompress_and_process_batch<
            t_vector_extension,
            blockwise_individual_f<
                    t_BlockSizeLog, t_format_selector, t_vector_extension
            >,
            t_op_vector,
            t_extra_args ...
    > {
        using t_ve = t_vector_extension;
        IMPORT_VECTOR_BOILER_PLATE(t_ve)
        
        using in_f = blockwise_individual_f<
                t_BlockSizeLog, t_format_selector, t_vector_extension
        >;
        
    public:
        static void apply(
                const uint8_t * & p_In8,
                size_t p_CountInLog,
                typename t_op_vector<t_ve, t_extra_args ...>::state_t & p_State
        ) {
            using namespace vectorlib;
            
            size_t countProcessed = 0;
            
            while(countProcessed < p_CountInLog) {
                // Find out which format is used for the current block.
                format_code formatCode = static_cast<format_code>(*p_In8++);
                
                // Skip the padding to find the compressed payload.
                p_In8 = create_aligned_ptr(p_In8, vector_size_byte::value);
                
                // Delegate the decompression to the used format.
                switch(formatCode) {
                    // Template for a single case.
                    #define BWI_CASE_DECOMPR_AND_PROCESS(formatCode, format) \
                        case format_code::formatCode: \
                            decompress_and_process_batch< \
                                    t_ve, format, t_op_vector, t_extra_args ... \
                            >::apply(p_In8, t_BlockSizeLog, p_State); \
                            break;
                    // Handling of all non-static_vbp formats.
                    BWI_CASE_DECOMPR_AND_PROCESS(uncompr, uncompr_f)
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultDynamicVbp, DEFAULT_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultGroupSimple, DEFAULT_GROUP_SIMPLE_F(t_ve))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultDeltaDynamicVbp, DEFAULT_DELTA_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultDeltaGroupSimple, DEFAULT_DELTA_GROUP_SIMPLE_F(t_ve))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultForDynamicVbp, DEFAULT_FOR_DYNAMIC_VBP_F(t_ve))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultForGroupSimple, DEFAULT_FOR_GROUP_SIMPLE_F(t_ve))
                    // Special treatment for static_vbp.
                    // Generated with Python:
                    // for bw in range(1, 64 + 1):
                    //   print("BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_{}, DEFAULT_STATIC_VBP_F(t_ve, {}))".format(bw, bw))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_1, DEFAULT_STATIC_VBP_F(t_ve, 1))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_2, DEFAULT_STATIC_VBP_F(t_ve, 2))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_3, DEFAULT_STATIC_VBP_F(t_ve, 3))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_4, DEFAULT_STATIC_VBP_F(t_ve, 4))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_5, DEFAULT_STATIC_VBP_F(t_ve, 5))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_6, DEFAULT_STATIC_VBP_F(t_ve, 6))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_7, DEFAULT_STATIC_VBP_F(t_ve, 7))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_8, DEFAULT_STATIC_VBP_F(t_ve, 8))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_9, DEFAULT_STATIC_VBP_F(t_ve, 9))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_10, DEFAULT_STATIC_VBP_F(t_ve, 10))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_11, DEFAULT_STATIC_VBP_F(t_ve, 11))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_12, DEFAULT_STATIC_VBP_F(t_ve, 12))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_13, DEFAULT_STATIC_VBP_F(t_ve, 13))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_14, DEFAULT_STATIC_VBP_F(t_ve, 14))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_15, DEFAULT_STATIC_VBP_F(t_ve, 15))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_16, DEFAULT_STATIC_VBP_F(t_ve, 16))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_17, DEFAULT_STATIC_VBP_F(t_ve, 17))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_18, DEFAULT_STATIC_VBP_F(t_ve, 18))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_19, DEFAULT_STATIC_VBP_F(t_ve, 19))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_20, DEFAULT_STATIC_VBP_F(t_ve, 20))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_21, DEFAULT_STATIC_VBP_F(t_ve, 21))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_22, DEFAULT_STATIC_VBP_F(t_ve, 22))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_23, DEFAULT_STATIC_VBP_F(t_ve, 23))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_24, DEFAULT_STATIC_VBP_F(t_ve, 24))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_25, DEFAULT_STATIC_VBP_F(t_ve, 25))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_26, DEFAULT_STATIC_VBP_F(t_ve, 26))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_27, DEFAULT_STATIC_VBP_F(t_ve, 27))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_28, DEFAULT_STATIC_VBP_F(t_ve, 28))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_29, DEFAULT_STATIC_VBP_F(t_ve, 29))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_30, DEFAULT_STATIC_VBP_F(t_ve, 30))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_31, DEFAULT_STATIC_VBP_F(t_ve, 31))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_32, DEFAULT_STATIC_VBP_F(t_ve, 32))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_33, DEFAULT_STATIC_VBP_F(t_ve, 33))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_34, DEFAULT_STATIC_VBP_F(t_ve, 34))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_35, DEFAULT_STATIC_VBP_F(t_ve, 35))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_36, DEFAULT_STATIC_VBP_F(t_ve, 36))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_37, DEFAULT_STATIC_VBP_F(t_ve, 37))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_38, DEFAULT_STATIC_VBP_F(t_ve, 38))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_39, DEFAULT_STATIC_VBP_F(t_ve, 39))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_40, DEFAULT_STATIC_VBP_F(t_ve, 40))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_41, DEFAULT_STATIC_VBP_F(t_ve, 41))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_42, DEFAULT_STATIC_VBP_F(t_ve, 42))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_43, DEFAULT_STATIC_VBP_F(t_ve, 43))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_44, DEFAULT_STATIC_VBP_F(t_ve, 44))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_45, DEFAULT_STATIC_VBP_F(t_ve, 45))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_46, DEFAULT_STATIC_VBP_F(t_ve, 46))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_47, DEFAULT_STATIC_VBP_F(t_ve, 47))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_48, DEFAULT_STATIC_VBP_F(t_ve, 48))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_49, DEFAULT_STATIC_VBP_F(t_ve, 49))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_50, DEFAULT_STATIC_VBP_F(t_ve, 50))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_51, DEFAULT_STATIC_VBP_F(t_ve, 51))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_52, DEFAULT_STATIC_VBP_F(t_ve, 52))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_53, DEFAULT_STATIC_VBP_F(t_ve, 53))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_54, DEFAULT_STATIC_VBP_F(t_ve, 54))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_55, DEFAULT_STATIC_VBP_F(t_ve, 55))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_56, DEFAULT_STATIC_VBP_F(t_ve, 56))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_57, DEFAULT_STATIC_VBP_F(t_ve, 57))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_58, DEFAULT_STATIC_VBP_F(t_ve, 58))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_59, DEFAULT_STATIC_VBP_F(t_ve, 59))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_60, DEFAULT_STATIC_VBP_F(t_ve, 60))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_61, DEFAULT_STATIC_VBP_F(t_ve, 61))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_62, DEFAULT_STATIC_VBP_F(t_ve, 62))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_63, DEFAULT_STATIC_VBP_F(t_ve, 63))
                    BWI_CASE_DECOMPR_AND_PROCESS(defaultStaticVbp_64, DEFAULT_STATIC_VBP_F(t_ve, 64))
                    // Undefine the case template.
                    #undef BWI_CASE_DECOMPR_AND_PROCESS
#ifdef BWI_CHECK_FORMAT_CODE
                    default:
                        throw std::runtime_error(
                                "unsupported format detected in "
                                "blockwise_individual_f decompression and "
                                "processing: " +
                                std::to_string(static_cast<uint8_t>(formatCode))
                        );
#endif
                }
                countProcessed += t_BlockSizeLog;
            }
        }
    };
    
}
#endif //MORPHSTORE_CORE_MORPHING_BLOCKWISE_INDIVIDUAL_H
