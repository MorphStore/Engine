/**
 * @file dummy_format_selectors.h
 * @brief Simple examples for naive format selectors, mainly to serve for
 * testing and as naive evaluation baselines.
 */

#ifndef MORPHSTORE_CORE_FORMAT_SELECTION_DUMMY_FORMAT_SELECTORS_H
#define MORPHSTORE_CORE_FORMAT_SELECTION_DUMMY_FORMAT_SELECTORS_H

#include <core/format_selection/format_selector_commons.h>
#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>

#include <chrono>
#include <random>

#include <cstdint>

namespace morphstore {
    
    template<format_code t_FormatCode>
    struct static_format_selector_helper {
        /**
         * @brief A format selector always choosing a fixed format.
         * 
         * Meant only for testing and as a simple baseline for evaluations. Not
         * meant to be effective.
         */
        template<class t_vector_extension>
        struct selector : public format_selector<t_vector_extension> {
            static format_code choose(
                    MSV_CXX_ATTRIBUTE_PPUNUSED
                    const typename t_vector_extension::base_t * p_In,
                    MSV_CXX_ATTRIBUTE_PPUNUSED
                    size_t p_CountLog
            ) {
                return t_FormatCode;
            }
        };
    };
    
    /**
     * @brief A format selector randomly choosing one out of a set of formats.
     * 
     * Meant for testing `blockwise_individual_f` without a sophisticated data
     * analysis and format selection. Not meant to be efficient or effective.
     */
    template<class t_vector_extension>
    struct random_format_selector : public format_selector<t_vector_extension> {
        static format_code choose(
                MSV_CXX_ATTRIBUTE_PPUNUSED
                const typename t_vector_extension::base_t * p_In,
                MSV_CXX_ATTRIBUTE_PPUNUSED size_t
                p_CountLog
        ) {
            // We do not allow static_vbp_f here, because it would require
            // determining the maximum bit width in the data. But we want it as
            // simple as possible here.
            std::default_random_engine gen(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count()
            );
            std::uniform_int_distribution<uint64_t> distr(0, 6);
            switch(distr(gen)) {
                case 1: return format_code::defaultDynamicVbp;
                case 2: return format_code::defaultGroupSimple;
                case 3: return format_code::defaultDeltaDynamicVbp;
                case 4: return format_code::defaultDeltaGroupSimple;
                case 5: return format_code::defaultForDynamicVbp;
                case 6: return format_code::defaultForGroupSimple;
                default: return format_code::uncompr;
            }
        }
    };
    
}
#endif //MORPHSTORE_CORE_FORMAT_SELECTION_DUMMY_FORMAT_SELECTORS_H