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
 * @file project_benchmark_test.cpp
 * @brief A testing and micro-benchmarking suite for multiple/all variants of
 * the project-operator.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/morph.h>
#include <core/morphing/static_vbp.h>
#include <core/operators/scalar/project_compr.h>
#include <core/operators/scalar/project_uncompr.h>
#include <core/operators/vectorized/project_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/equality_check.h>
#include <core/utils/helper.h>
#include <core/utils/math.h>
#include <core/utils/monitoring.h>
#include <core/utils/preprocessor.h>
#include <core/utils/printing.h>
#include <core/utils/processing_style.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <map>
#include <vector>

using namespace morphstore;

#ifdef MSV_USE_MONITORING
const std::string colInDataCount    ("inDataCount");
const std::string colInPosCount     ("inPosCount");
const std::string colProcessingStyle("processingStyle");
const std::string colInPosFormat    ("inPosFormat");
const std::string colRuntime        ("runtime[µs]");
const std::string colCheck          ("check");
#endif

/**
 * @brief A wrapper for calling a template specialization of the
 * project-operator.
 * 
 * Necessary, since different template specializations can differ in their
 * columns' formats, such that there is no common function pointer type for
 * calling them.
 * 
 * This function morphs the uncompressed input positions to the format
 * expected by the variant and executes the variant.
 * 
 * This function has the following template parameters:
 * - `t_ps_morph` The processing style to use for the morph operator. Note that
 *   this does not matter if `t_in_pos_f` is uncompressed.
 * - `t_ps_project` The processing style to use for the actual
 *   project-operator.
 * - `t_in_pos_f` The (un)compressed format of the project-operator's input
 *   positions column.
 * 
 * @param p_InDataCol The project-operator's uncompressed input data column.
 * @param p_InPosCol The project-operator's uncompressed input positions
 * column.
 * @param p_ProcessingStyle The name of the processing style to use.
 * @param p_InPosFormat The name of the input positions format to use.
 * @return The project-operator's uncompressed output data column.
 */
template<
        processing_style_t t_ps_morph,
        processing_style_t t_ps_project,
        class t_in_pos_f
>
const column<uncompr_f> *
measure_project(
        const column<uncompr_f> * const p_InDataCol,
        const column<uncompr_f> * const p_InPosCol,
        MSV_CXX_ATTRIBUTE_PPUNUSED const std::string & p_ProcessingStyle,
        MSV_CXX_ATTRIBUTE_PPUNUSED const std::string & p_InPosFormat
) {
    // Note that this is a no-op if t_in_pos_f is uncompr_f, i.e., if the
    // operator variant expects its input positions in the uncompressed format.
    auto inPosColMorphed = morph<t_ps_morph, t_in_pos_f>(p_InPosCol);
    
    // Execution of the operator, wrapped into runtime measurement.
#ifdef MSV_USE_MONITORING
    const size_t inDataCount = p_InDataCol->get_count_values();
    const size_t inPosCount = p_InPosCol->get_count_values();
#endif
    MONITORING_START_INTERVAL_FOR(
        colRuntime, inDataCount, inPosCount, p_ProcessingStyle, p_InPosFormat
    );
    auto outPosCol = project<t_ps_project, uncompr_f, uncompr_f, t_in_pos_f>(
            p_InDataCol, inPosColMorphed
    );
    MONITORING_END_INTERVAL_FOR(
        colRuntime, inDataCount, inPosCount, p_ProcessingStyle, p_InPosFormat
    );
            
    // The morphed input positions can be freed if the morph-operator was a
    // no-op.
    if(!std::is_same<t_in_pos_f, uncompr_f>::value)
        delete inPosColMorphed;
    
    return outPosCol;
}

/**
 * @brief A function pointer type for (the above wrapper of) the
 * project-operator.
 */
typedef const column<uncompr_f> *
(*project_fp_t) (
        const column<uncompr_f> * const,
        const column<uncompr_f> * const,
        const std::string &,
        const std::string &
);

// @todo Not sure if macros can be documented this way.
/**
 * @brief Expands to an initializer list for a tuple consisting of a function
 * pointer to (the wrapper of) the specified project-operator variant and
 * textual representations of the processing style and input position column
 * format to use.
 * 
 * This macro's parameters are the template parameters of the function
 * `measure_project` above.
 */
#define MAKE_VARIANT(t_ps_morph, t_ps_project, t_in_pos_f) { \
    &measure_project< \
            processing_style_t::t_ps_morph, \
            processing_style_t::t_ps_project, \
            t_in_pos_f \
    >, \
    #t_ps_project, \
    STR_EVAL_MACROS(t_in_pos_f) \
}

typedef std::tuple<
    project_fp_t, const std::string, const std::string
> variant_t;
    
// This must be a macro, because if it is a C++ variable/constant, then the
// output will always contain the identifier/expression, but not the value.
// The value must be a literal here, it can be calculated as
// sizeof(__m128i) / sizeof(uint64_t) .
#define STEP_128 2

/**
 * @brief Tests if all considered variants of the project-operator yield
 * exactly the same output in the setting specified by the parameters.
 * 
 * The project-operator's input columns are generated. All operator variants
 * are executed on these columns. The runtimes are measured and some details
 * regarding progress, correctness checks and measurements are printed.
 * 
 * @param p_InDataCount The number of data elements in the project-operator's
 * input data column.
 * @param p_InPosCount The number of data elements in the project-operator's
 * input positions column.
 * @return `true` if all variants yielded exactly the same output, `false`
 * otherwise.
 */
bool evaluate_variants(size_t p_InDataCount, size_t p_InPosCount) {
    std::cerr
            << "Setting" << std::endl
            << "\tParameters" << std::endl
            << "\t\tinDataCount:\t " << p_InDataCount << std::endl
            << "\t\tinPosCount:\t " << p_InPosCount << std::endl;
    
    // ------------------------------------------------------------------------
    // Data generation.
    // ------------------------------------------------------------------------
    std::cerr
            << "\tData generation" << std::endl
            << "\t\tstarted... ";
    std::cerr.flush();
    auto inDataCol = generate_with_distr(
            p_InDataCount,
            std::uniform_int_distribution<uint64_t>(123, 456),
            false
    );
    auto inPosCol = generate_with_distr(
            p_InPosCount,
            std::uniform_int_distribution<uint64_t>(0, p_InDataCount - 1),
            false
    );
    std::cerr << "done" << std::endl;
    
    // ------------------------------------------------------------------------
    // Variant specification.
    // ------------------------------------------------------------------------
    // Each variant is a pair. The first element is a pointer to the function
    // to call, the second element is the name of the variant (to be used as
    // the monitoring counter name).
    std::vector<variant_t> variants = {
        MAKE_VARIANT(scalar, scalar, uncompr_f),
        MAKE_VARIANT(scalar, vec128, uncompr_f),
        MAKE_VARIANT(scalar, vec256, uncompr_f)
    };
    // These variants can only be executed if their static bit width is high
    // enough to represent the highest possible position.
    std::map<unsigned, std::vector<variant_t> > staticVPBVariantsByBw = {
        // Generated with Python:
        // for bw in range(1, 64+1):
        //   print("{{{: >2}, {{MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<{: >2}, STEP_128>))}}}}{}".format(bw, bw, "," if bw < 64 else ""))
        { 1, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 1, STEP_128>))}},
        { 2, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 2, STEP_128>))}},
        { 3, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 3, STEP_128>))}},
        { 4, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 4, STEP_128>))}},
        { 5, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 5, STEP_128>))}},
        { 6, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 6, STEP_128>))}},
        { 7, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 7, STEP_128>))}},
        { 8, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 8, STEP_128>))}},
        { 9, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f< 9, STEP_128>))}},
        {10, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<10, STEP_128>))}},
        {11, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<11, STEP_128>))}},
        {12, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<12, STEP_128>))}},
        {13, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<13, STEP_128>))}},
        {14, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<14, STEP_128>))}},
        {15, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<15, STEP_128>))}},
        {16, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<16, STEP_128>))}},
        {17, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<17, STEP_128>))}},
        {18, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<18, STEP_128>))}},
        {19, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<19, STEP_128>))}},
        {20, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<20, STEP_128>))}},
        {21, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<21, STEP_128>))}},
        {22, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<22, STEP_128>))}},
        {23, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<23, STEP_128>))}},
        {24, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<24, STEP_128>))}},
        {25, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<25, STEP_128>))}},
        {26, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<26, STEP_128>))}},
        {27, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<27, STEP_128>))}},
        {28, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<28, STEP_128>))}},
        {29, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<29, STEP_128>))}},
        {30, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<30, STEP_128>))}},
        {31, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<31, STEP_128>))}},
        {32, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<32, STEP_128>))}},
        {33, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<33, STEP_128>))}},
        {34, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<34, STEP_128>))}},
        {35, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<35, STEP_128>))}},
        {36, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<36, STEP_128>))}},
        {37, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<37, STEP_128>))}},
        {38, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<38, STEP_128>))}},
        {39, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<39, STEP_128>))}},
        {40, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<40, STEP_128>))}},
        {41, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<41, STEP_128>))}},
        {42, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<42, STEP_128>))}},
        {43, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<43, STEP_128>))}},
        {44, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<44, STEP_128>))}},
        {45, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<45, STEP_128>))}},
        {46, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<46, STEP_128>))}},
        {47, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<47, STEP_128>))}},
        {48, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<48, STEP_128>))}},
        {49, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<49, STEP_128>))}},
        {50, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<50, STEP_128>))}},
        {51, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<51, STEP_128>))}},
        {52, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<52, STEP_128>))}},
        {53, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<53, STEP_128>))}},
        {54, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<54, STEP_128>))}},
        {55, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<55, STEP_128>))}},
        {56, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<56, STEP_128>))}},
        {57, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<57, STEP_128>))}},
        {58, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<58, STEP_128>))}},
        {59, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<59, STEP_128>))}},
        {60, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<60, STEP_128>))}},
        {61, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<61, STEP_128>))}},
        {62, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<62, STEP_128>))}},
        {63, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<63, STEP_128>))}},
        {64, {MAKE_VARIANT(vec128, scalar, SINGLE_ARG(static_vbp_f<64, STEP_128>))}}
    };

    // Add all operator variants using the static_vbp format with a high enough
    // bit width.
    const unsigned minBw = effective_bitwidth(p_InDataCount - 1);
    for(const auto & vs : staticVPBVariantsByBw)
        if(vs.first >= minBw)
            for(const auto & v : vs.second)
                variants.push_back(v);
    
    // Just for the sake of a nicely formatted output.
    size_t maxLen = 0;
    for(auto v : variants) {
        const size_t len = std::get<1>(v).size() + 2 + std::get<2>(v).size();
        if(len > maxLen)
            maxLen = len;
    }
    
    // ------------------------------------------------------------------------
    // Variant execution.
    // ------------------------------------------------------------------------
    // This column will be used as the reference for correctness.
    const column<uncompr_f> * outDataCol_ref = nullptr;
    // Whether so far all variants yielded exactly the same output.
    bool allGood = true;
    std::cerr
            << "\tVariant execution" << std::endl;
    for(auto v : variants) {
        const std::string & vProcessingStyle = std::get<1>(v);
        const std::string & vInPosFormat = std::get<2>(v);
        
        MONITORING_CREATE_MONITOR(
                MONITORING_MAKE_MONITOR(
                        p_InDataCount,
                        p_InPosCount,
                        vProcessingStyle,
                        vInPosFormat
                ),
                MONITORING_KEY_IDENTS(
                        colInDataCount,
                        colInPosCount,
                        colProcessingStyle,
                        colInPosFormat
                )
        );
        std::cerr
                << "\t\t" << std::setw(maxLen) << std::left
                << (vProcessingStyle + ", " + vInPosFormat) << " started... ";
        std::cerr.flush();
        // Calling the function of the variant.
        auto outDataCol = (*(std::get<0>(v)))(
                inDataCol, inPosCol, vProcessingStyle, vInPosFormat
        );
        std::cerr << "done -> ";
        if(outDataCol_ref == nullptr) {
            // The first variant serves as the reference (w.r.t. correctness)
            // for all following variants. Thus, we have to keep its output
            // till the end.
            outDataCol_ref = outDataCol;
            std::cerr << "reference" << std::endl;
            MONITORING_ADD_INT_FOR(
                    colCheck, -1,
                    p_InDataCount, p_InPosCount, vProcessingStyle, vInPosFormat
            );
        }
        else {
            // All following variants are compared to the reference.
            equality_check ec(outDataCol_ref, outDataCol);
            // Now the output of the current variant is not required any more.
            delete outDataCol;
            // Print if this variant yielded exactly the same output as the
            // reference.
            const bool good = ec.good();
            std::cerr << equality_check::ok_str(good) << std::endl;
            if(!good)
                std::cerr
                        << std::endl
                        << "\t\t\tdetails:" << std::endl
                        << ec << std::endl;
            allGood = allGood && good;
            MONITORING_ADD_INT_FOR(
                    colCheck, good,
                    p_InDataCount, p_InPosCount, vProcessingStyle, vInPosFormat
            );
        }
    }
    // Now we can free the reference output.
    delete outDataCol_ref;
    
    // Free the generated input data.
    delete inDataCol;
    delete inPosCol;
    
    return allGood;
}

#undef STEP_128

int main(void) {
#ifdef MSV_NO_SELFMANAGED_MEMORY
    bool curGood;
    bool allGood = true;
    
    // Setting in which a few positions refer to a large data column.
    curGood = evaluate_variants(128 * 1000 * 1000, 16 * 1024);
    allGood = allGood && curGood;
    // Setting in which sequentially reading the input position column should
    // have a significant impact.
    curGood = evaluate_variants(2, 128 * 1000 * 1000);
    allGood = allGood && curGood;
    // Setting in which sequentially reading the input position column should
    // not have a significant impact.
    curGood = evaluate_variants(1000 * 1000, 128 * 1000);
    allGood = allGood && curGood;
    
    // Summary.
    std::cerr
            << "Summary" << std::endl
            << '\t' << (allGood ? "all ok" : "some NOT OK") << std::endl;
    
    // Output of the runtimes.
    MONITORING_PRINT_MONITORS(monitorCsvLog);
    
    return !allGood;
#else
    // @todo Make it work with self-managed memory.
    std::cerr
            << "Currently, this benchmark only works with non-self-managed "
            << "memory. Compile MorphStore with build.sh -noSelfManaging ."
            << std::endl;
    return 1;
#endif
}