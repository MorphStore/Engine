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
#include <core/operators/scalar/project_uncompr.h>
#include <core/operators/vectorized/project_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/equality_check.h>
#include <core/utils/monitoring.h>
#include <core/utils/printing.h>
#include <core/utils/processing_style.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

using namespace morphstore;

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
 * @param p_CounterName The name of the monitoring counter to use.
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
        const std::string & p_CounterName
) {
    // Note that this is a no-op if t_in_pos_f is uncompr_f, i.e., if the
    // operator variant expects its input positions in the uncompressed format.
    auto inPosColMorphed = morph<t_ps_morph, t_in_pos_f>(p_InPosCol);
    
#ifndef MSV_USE_MONITORING
    // @todo This is ugly.
    // p_CounterName would only be used if monitoring is enabled. To prevent a
    // compiler error due to an unused variable, we must do something with it
    // even if monitoring is disabled.
    std::cout
            << "(the monitoring counter's name would have been '"
            << p_CounterName << "') ";
#endif
    
    // Execution of the operator, wrapped into runtime measurement.
    MONITOR_START_INTERVAL(p_CounterName)
    auto outPosCol = project<t_ps_project, uncompr_f, uncompr_f, t_in_pos_f>(
            p_InDataCol, inPosColMorphed
    );
    MONITOR_END_INTERVAL(p_CounterName)
            
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
        const std::string &
);

// @todo Not sure if macros can be documented this way.
/**
 * @brief Expands to an initializer list for a pair consisting of a function
 * pointer to (the wrapper of) the specified project-operator variant and a
 * suitable textual representation, i.e., name, of the variant.
 * 
 * This macro's parameters are the template parameters of the function
 * `measure_project` above.
 */
#define PTR_AND_NAME(t_ps_morph, t_ps_project, t_in_pos_f) { \
    &measure_project< \
            processing_style_t::t_ps_morph, \
            processing_style_t::t_ps_project, \
            t_in_pos_f \
    >, \
    #t_ps_project ", " #t_in_pos_f \
}

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
    std::cout
            << "Setting" << std::endl
            << "\tParameters" << std::endl
            << "\t\tinDataCount:\t " << p_InDataCount << std::endl
            << "\t\tinPosCount:\t " << p_InPosCount << std::endl;
    
    // ------------------------------------------------------------------------
    // Data generation.
    // ------------------------------------------------------------------------
    std::cout
            << "\tData generation" << std::endl
            << "\t\tstarted... ";
    std::cout.flush();
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
    std::cout << "done" << std::endl;
    
    // ------------------------------------------------------------------------
    // Variant specification.
    // ------------------------------------------------------------------------
    // Each variant is a pair. The first element is a pointer to the function
    // to call, the second element is the name of the variant (to be used as
    // the monitoring counter name).
    std::vector<std::pair<project_fp_t, std::string> > variants = {
        PTR_AND_NAME(scalar, scalar, uncompr_f),
        PTR_AND_NAME(scalar, vec128, uncompr_f),
        PTR_AND_NAME(scalar, vec256, uncompr_f)
    };
    // Just for the sake of a nicely formatted output.
    size_t maxLen = 0;
    for(auto v : variants) {
        const size_t len = v.second.length();
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
    std::cout
            << "\tVariant execution" << std::endl;
    for(auto v : variants) {
        std::cout
                << "\t\t" << std::setw(maxLen) << std::left << v.second
                << " started... "; 
        std::cout.flush();
        // Calling the function of the variant.
        auto outDataCol = (*(v.first))(inDataCol, inPosCol, v.second);
        std::cout << "done -> ";
        if(outDataCol_ref == nullptr) {
            // The first variant serves as the reference (w.r.t. correctness)
            // for all following variants. Thus, we have to keep its output
            // till the end.
            outDataCol_ref = outDataCol;
            std::cout << "reference" << std::endl;
        }
        else {
            // All following variants are compared to the reference.
            equality_check ec(outDataCol_ref, outDataCol);
            // Now the output of the current variant is not required any more.
            delete outDataCol;
            // Print if this variant yielded exactly the same output as the
            // reference.
            const bool good = ec.good();
            std::cout << equality_check::ok_str(good) << std::endl;
            if(!good)
                std::cout
                        << std::endl
                        << "\t\t\tdetails:" << std::endl
                        << ec << std::endl;
            allGood = allGood && good;
        }
    }
    // Now we can free the reference output.
    delete outDataCol_ref;
    
    // Free the generated input data.
    delete inDataCol;
    delete inPosCol;
    
    return allGood;
}

int main(void) {
#ifdef MSV_NO_SELFMANAGED_MEMORY
    bool allGood = true;
    
    // Setting in which a few positions refer to a large data column.
    allGood = allGood && evaluate_variants(128 * 1000 * 1000, 16 * 1024);
    // Setting in which sequentially reading the input position column should
    // have a significant impact.
    allGood = allGood && evaluate_variants(2, 128 * 1000 * 1000);
    // Setting in which sequentially reading the input position column should
    // not have a significant impact.
    allGood = allGood && evaluate_variants(1000 * 1000, 128 * 1000);
    
    // Output of the runtimes.
    // @todo These should be listed above, but the monitoring does not support
    // this at the moment.
    std::cout << "Runtimes [µs]" << std::endl;
    MONITOR_PRINT_ALL(monitorShellLog, true)
    
    // Summary.
    std::cout
            << "Summary" << std::endl
            << '\t' << (allGood ? "all ok" : "some NOT OK") << std::endl;
    
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