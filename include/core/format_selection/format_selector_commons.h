/**
 * @file format_selector_commons.h
 * @brief Fundamental things for format selection.
 */

#ifndef MORPHSTORE_CORE_FORMAT_SELECTION_FORMAT_SELECTOR_COMMONS_H
#define MORPHSTORE_CORE_FORMAT_SELECTION_FORMAT_SELECTOR_COMMONS_H

#include <core/utils/basic_types.h>

namespace morphstore {
    
    /**
     * @brief A numerical representation of a format.
     * 
     * This is only intended to be used for representing formats at run-time,
     * especially during the format selection and `blockwise_individual_f`.
     * 
     * The prefix `default` refers to the default template parameters of the
     * respective format given a particular processing style (see
     * `core/morphing/default_formats.h`). In other words, it is currently not
     * supported to choose even the template parameters of a format, we simply
     * assume the defaults for the given processing style.
     */
    enum class format_code {
        uncompr = 0,
        defaultDynamicVbp = 65,
        defaultGroupSimple,
        defaultDeltaDynamicVbp,
        defaultDeltaGroupSimple,
        defaultForDynamicVbp,
        defaultForGroupSimple,
        // The codes 1 to 64 represent the default static_vbp_f for the
        // respective bit width by convention. Several places in the code make
        // use of this, so take care should you want to change that.
        // Generated with Python:
        // for bw in range(1, 64 + 1):
        //   print("defaultStaticVbp_{} = {},".format(bw, bw))
        defaultStaticVbp_1 = 1,
        defaultStaticVbp_2 = 2,
        defaultStaticVbp_3 = 3,
        defaultStaticVbp_4 = 4,
        defaultStaticVbp_5 = 5,
        defaultStaticVbp_6 = 6,
        defaultStaticVbp_7 = 7,
        defaultStaticVbp_8 = 8,
        defaultStaticVbp_9 = 9,
        defaultStaticVbp_10 = 10,
        defaultStaticVbp_11 = 11,
        defaultStaticVbp_12 = 12,
        defaultStaticVbp_13 = 13,
        defaultStaticVbp_14 = 14,
        defaultStaticVbp_15 = 15,
        defaultStaticVbp_16 = 16,
        defaultStaticVbp_17 = 17,
        defaultStaticVbp_18 = 18,
        defaultStaticVbp_19 = 19,
        defaultStaticVbp_20 = 20,
        defaultStaticVbp_21 = 21,
        defaultStaticVbp_22 = 22,
        defaultStaticVbp_23 = 23,
        defaultStaticVbp_24 = 24,
        defaultStaticVbp_25 = 25,
        defaultStaticVbp_26 = 26,
        defaultStaticVbp_27 = 27,
        defaultStaticVbp_28 = 28,
        defaultStaticVbp_29 = 29,
        defaultStaticVbp_30 = 30,
        defaultStaticVbp_31 = 31,
        defaultStaticVbp_32 = 32,
        defaultStaticVbp_33 = 33,
        defaultStaticVbp_34 = 34,
        defaultStaticVbp_35 = 35,
        defaultStaticVbp_36 = 36,
        defaultStaticVbp_37 = 37,
        defaultStaticVbp_38 = 38,
        defaultStaticVbp_39 = 39,
        defaultStaticVbp_40 = 40,
        defaultStaticVbp_41 = 41,
        defaultStaticVbp_42 = 42,
        defaultStaticVbp_43 = 43,
        defaultStaticVbp_44 = 44,
        defaultStaticVbp_45 = 45,
        defaultStaticVbp_46 = 46,
        defaultStaticVbp_47 = 47,
        defaultStaticVbp_48 = 48,
        defaultStaticVbp_49 = 49,
        defaultStaticVbp_50 = 50,
        defaultStaticVbp_51 = 51,
        defaultStaticVbp_52 = 52,
        defaultStaticVbp_53 = 53,
        defaultStaticVbp_54 = 54,
        defaultStaticVbp_55 = 55,
        defaultStaticVbp_56 = 56,
        defaultStaticVbp_57 = 57,
        defaultStaticVbp_58 = 58,
        defaultStaticVbp_59 = 59,
        defaultStaticVbp_60 = 60,
        defaultStaticVbp_61 = 61,
        defaultStaticVbp_62 = 62,
        defaultStaticVbp_63 = 63,
        defaultStaticVbp_64 = 64,
    };
    
    /**
     * @brief Superclass of all classes capable of selecting a format for some
     * given data.
     * 
     * We do not use a simple function, but a class with a static member
     * function here, because:
     * - we want to be able to pass a format selector as a template parameter
     *   (see `blockwise_individual_f`)
     * - we want to use the inheritance relation in static assertions
     * 
     * A format selector must know the processing style, because:
     * - the processing style generally has an impact on the behavior of the
     *   vectorized (de)compression and/or processing of the data
     * - a format selector usually needs to analyze the given data to extract
     *   some relevant data characteristics, which should also be done in a
     *   vectorized way for efficiency (although this could be decoupled from
     *   the processing style assumed for the formats)
     */
    template<class t_vector_extension>
    struct format_selector {
        /**
         * @brief Selects a format for a given uncompressed data buffer.
         * 
         * @param p_In The *uncompressed* data for which a format shall be
         * selected.
         * @param p_CountLog The number of logical data elements in the given
         * data buffer.
         * @return A numerical format code representing the selected format.
         */
        static format_code choose(
                const typename t_vector_extension::base_t * p_In,
                size_t p_CountLog
        ) = delete;
    };
    
}
#endif //MORPHSTORE_CORE_FORMAT_SELECTION_FORMAT_SELECTOR_COMMONS_H