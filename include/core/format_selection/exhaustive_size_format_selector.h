/**
 * @file exhaustive_size_format_selector.h
 * @brief A format selector choosing the format actually minimizing (or
 * maximizing, if desired) the compressed size.
 */

#ifndef MORPHSTORE_CORE_FORMAT_SELECTION_EXHAUSTIVE_SIZE_FORMAT_SELECTOR_H
#define MORPHSTORE_CORE_FORMAT_SELECTION_EXHAUSTIVE_SIZE_FORMAT_SELECTOR_H

#include <core/format_selection/format_selector_commons.h>
#include <core/memory/management/utils/alignment_helper.h>
#include <core/morphing/blockwise_individual.h>
#include <core/morphing/default_formats.h>
#include <core/morphing/delta.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/for.h>
#include <core/morphing/format.h>
#include <core/morphing/group_simple.h>
#include <core/morphing/morph_batch.h>
#include <core/morphing/uncompr.h>
#include <core/utils/basic_types.h>
#include <core/utils/data_properties.h>
#include <core/utils/math.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>

#include <limits>

#include <cstdint>

namespace morphstore {

    // If t_ChooseBest is true, the format yielding the best size will be
    // selected, otherwise the format yielding the worst size.
    template<bool t_ChooseBest>
    struct exhaustive_size_format_selector_helper {
        /**
         * @brief A format selector choosing the format actually minimizing (or
         * maximizing, if desired) the compressed size.
         * 
         * Useful as a baseline for what could be achieved with respect to the
         * compressed size in the best (or worst) case.
         * 
         * The current implementation exhaustively tries the compression to
         * each supported format and chooses the one yielding the minimum (or
         * maximum) size. It it not meant to be efficient regarding
         * performance. Nevertheless, when parameterized to choose the best
         * (not the worst) size, it is very effective regarding compressed
         * size.
         */
        template<class t_vector_extension>
        class selector : public format_selector<t_vector_extension> {

            template<class t_target_f, format_code t_FormatCode>
            MSV_CXX_ATTRIBUTE_INLINE
            static void better_size_and_format(
                    const uint8_t * p_In8,
                    size_t p_CountLog,
                    uint8_t * p_TmpOut8,
                    size_t & p_BestSizeSoFar,
                    format_code & p_BestFormatSoFar
            ) {
                const uint8_t * const initTmpOut8 = p_TmpOut8;
                morph_batch<t_vector_extension, t_target_f, uncompr_f>(
                        p_In8, p_TmpOut8, p_CountLog
                );
                const size_t sizeByte = p_TmpOut8 - initTmpOut8;
                if(t_ChooseBest == (sizeByte < p_BestSizeSoFar)) {
                    p_BestSizeSoFar = sizeByte;
                    p_BestFormatSoFar = t_FormatCode;
                }
            }

        public:
            static format_code choose(
                    const typename t_vector_extension::base_t * p_In,
                    size_t p_CountLog
            ) {
                using t_ve = t_vector_extension;
                IMPORT_VECTOR_BOILER_PLATE(t_ve)

                // @todo This might be expensive, and actually it does not need
                // to be repeated each time.
                const size_t sizeAlloc = blockwise_individual_helper::get_size_max_byte_single_block<t_ve>(p_CountLog);

                // Allocate a temporary buffer for compressing the data.
#ifdef MSV_NO_SELFMANAGED_MEMORY
                uint8_t * tmp8Unaligned = reinterpret_cast<uint8_t *>(
                        malloc(get_size_with_alignment_padding(sizeAlloc))
                );
                uint8_t * tmp8 = create_aligned_ptr(tmp8Unaligned);
#else
                uint8_t * tmp8 = malloc(sizeAlloc);
#endif

                const uint8_t * in8 = reinterpret_cast<const uint8_t *>(p_In);

                // Start with static_vbp if the size shall be optimized and it
                // is beneficial w.r.t. the compressed size, otherwise start
                // with the uncompressed format.
                size_t bestSizeSoFar;
                format_code bestFormatSoFar;
                const unsigned maxBw = determine_max_bitwidth<t_ve>(p_In, p_CountLog);
                if(t_ChooseBest && maxBw < std::numeric_limits<uint64_t>::digits) {
                    // The size when using static_vbp can easily be calculated,
                    // no need to execute the compression algorithm.
                    bestSizeSoFar = (maxBw * p_CountLog) / bitsPerByte;
                    // By convention (see format_code), the bit width itself
                    // is the format code of static_vbp with the respective bit
                    // width. 
                    bestFormatSoFar = static_cast<format_code>(maxBw);
                }
                else {
                    bestSizeSoFar = uncompr_f::get_size_max_byte(p_CountLog);
                    bestFormatSoFar = format_code::uncompr;
                }

                // Execute the compression algorithm of all other formats to
                // find out if any of them yields a better size.
                #define ESFS_UPDATE_BEST(formatCode, format) \
                    better_size_and_format<format, format_code::formatCode>( \
                            in8, p_CountLog, tmp8, bestSizeSoFar, bestFormatSoFar \
                    );
                ESFS_UPDATE_BEST(defaultDynamicVbp, DEFAULT_DYNAMIC_VBP_F(t_ve))
                ESFS_UPDATE_BEST(defaultGroupSimple, DEFAULT_GROUP_SIMPLE_F(t_ve))
                ESFS_UPDATE_BEST(defaultDeltaDynamicVbp, DEFAULT_DELTA_DYNAMIC_VBP_F(t_ve))
                ESFS_UPDATE_BEST(defaultDeltaGroupSimple, DEFAULT_DELTA_GROUP_SIMPLE_F(t_ve))
                ESFS_UPDATE_BEST(defaultForDynamicVbp, DEFAULT_FOR_DYNAMIC_VBP_F(t_ve))
                ESFS_UPDATE_BEST(defaultForGroupSimple, DEFAULT_FOR_GROUP_SIMPLE_F(t_ve))
                #undef ESFS_UPDATE_BEST

#ifdef MSV_NO_SELFMANAGED_MEMORY
                free(tmp8Unaligned);
#else
                // @todo The memory manager does not support freeing yet...
#endif

                return bestFormatSoFar;
            }
        };
    };
    
}
#endif //MORPHSTORE_CORE_FORMAT_SELECTION_EXHAUSTIVE_SIZE_FORMAT_SELECTOR_H
