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
 * @file operator_test_frames.h
 * @brief Functions for small tests of operators on uncompressed data.
 * @todo Reduce the code duplication.
 */

#ifndef MORPHSTORE_CORE_OPERATORS_OPERATOR_TEST_FRAMES_H
#define MORPHSTORE_CORE_OPERATORS_OPERATOR_TEST_FRAMES_H

#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/equality_check.h>
#include <core/utils/printing.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

namespace morphstore {
    
// ****************************************************************************
// Typedefs for operator-functions consuming one or two columns (and perhaps an
// integer) and producing one or two columns.
// ****************************************************************************
    
typedef const column<uncompr_f> * (*op_1in_1out)(
        const column<uncompr_f> *
);
typedef const column<uncompr_f> * (*op_1in_1out_2val)(
        const column<uncompr_f> *,
        const uint64_t,
        const size_t
);
typedef const column<uncompr_f> * (*op_2in_1out)(
        const column<uncompr_f> *,
        const column<uncompr_f> *
);
typedef const column<uncompr_f> * (*op_2in_1out_1val)(
        const column<uncompr_f> *,
        const column<uncompr_f> *,
        size_t
);
typedef const std::tuple<
        const column<uncompr_f> *,
        const column<uncompr_f> *
>
(*op_1in_2out_1val)(
        const column<uncompr_f> *,
        size_t
);
typedef const std::tuple<
        const column<uncompr_f> *,
        const column<uncompr_f> *
>
(*op_2in_2out_1val)(
        const column<uncompr_f> *,
        const column<uncompr_f> *,
        size_t
);

// ****************************************************************************
// Helper functions
// ****************************************************************************

void print_header(const std::string & opName) {
    std::cout << opName << "-operator test" << std::endl << std::endl;
}

void print_check(const std::string & outColName, const equality_check & ec) {
    std::cout
            << "Checking result column " << outColName << std::endl
            << std::endl << ec << std::endl;
}

void print_overall(bool allGood) {
    std::cout
            << "*****************" << std::endl
            << "* Overall: " << equality_check::ok_str(allGood) << std::endl
            << "*****************" << std::endl
            << std::endl;
}

// ****************************************************************************
// Test functions for each operator signature
// ****************************************************************************

bool test_op_1in_1out(
        const std::string & opName,
        op_1in_1out op,
        const column<uncompr_f> * inCol0,
        const std::string & inCol0Name,
        const column<uncompr_f> * outCol0Exp,
        const std::string & outCol0Name
) {
    print_header(opName);
    
    auto outCol0Fnd = op(inCol0);
    
    print_columns(
            print_buffer_base::decimal,
            inCol0, outCol0Fnd, outCol0Exp,
            inCol0Name,
            outCol0Name + " (found)", outCol0Name + " (expected)"
    );
    
    const equality_check ec0(outCol0Exp, outCol0Fnd);
    const bool allGood = ec0.good();
    
    std::cout << std::endl;
    print_check(outCol0Name, ec0);
    print_overall(allGood);
    
    return allGood;
}

bool test_op_1in_1out_2val(
        const std::string & opName,
        op_1in_1out_2val op,
        const column<uncompr_f> * inCol0,
        const std::string & inCol0Name,
        const column<uncompr_f> * outCol0Exp,
        const std::string & outCol0Name,
        uint64_t val0,
        size_t val1
) {
    print_header(opName);
    
    auto outCol0Fnd = op(inCol0, val0, val1);
    
    print_columns(
            print_buffer_base::decimal,
            inCol0, outCol0Fnd, outCol0Exp,
            inCol0Name,
            outCol0Name + " (found)", outCol0Name + " (expected)"
    );
    
    const equality_check ec0(outCol0Exp, outCol0Fnd);
    const bool allGood = ec0.good();
    
    std::cout << std::endl;
    print_check(outCol0Name, ec0);
    print_overall(allGood);
    
    return allGood;
}

bool test_op_2in_1out(
        const std::string & opName,
        op_2in_1out op,
        const column<uncompr_f> * inCol0,
        const column<uncompr_f> * inCol1,
        const std::string & inCol0Name,
        const std::string & inCol1Name,
        const column<uncompr_f> * outCol0Exp,
        const std::string & outCol0Name
) {
    print_header(opName);
    
    auto outCol0Fnd = op(inCol0, inCol1);
    
    print_columns(
            print_buffer_base::decimal,
            inCol0, inCol1, outCol0Fnd, outCol0Exp,
            inCol0Name, inCol1Name,
            outCol0Name + " (found)", outCol0Name + " (expected)"
    );
    
    const equality_check ec0(outCol0Exp, outCol0Fnd);
    const bool allGood = ec0.good();
    
    std::cout << std::endl;
    print_check(outCol0Name, ec0);
    print_overall(allGood);
    
    return allGood;
}

bool test_op_2in_1out_1val(
        const std::string & opName,
        op_2in_1out_1val op,
        const column<uncompr_f> * inCol0,
        const column<uncompr_f> * inCol1,
        const std::string & inCol0Name,
        const std::string & inCol1Name,
        const column<uncompr_f> * outCol0Exp,
        const std::string & outCol0Name,
        size_t val0
) {
    print_header(opName);
    
    auto outCol0Fnd = op(inCol0, inCol1, val0);
    
    print_columns(
            print_buffer_base::decimal,
            inCol0, inCol1, outCol0Fnd, outCol0Exp,
            inCol0Name, inCol1Name,
            outCol0Name + " (found)", outCol0Name + " (expected)"
    );
    
    const equality_check ec0(outCol0Exp, outCol0Fnd);
    const bool allGood = ec0.good();
    
    std::cout << std::endl;
    print_check(outCol0Name, ec0);
    print_overall(allGood);
    
    return allGood;
}

bool test_op_1in_2out_1val(
        const std::string & opName,
        op_1in_2out_1val op,
        const column<uncompr_f> * inCol0,
        const std::string & inCol0Name,
        const column<uncompr_f> * outCol0Exp,
        const column<uncompr_f> * outCol1Exp,
        const std::string & outCol0Name,
        const std::string & outCol1Name,
        size_t val0
) {
    print_header(opName);
    
    const column<uncompr_f> * outCol0Fnd;
    const column<uncompr_f> * outCol1Fnd;
    std::tie(outCol0Fnd, outCol1Fnd) = op(inCol0, val0);
    
    print_columns(
            print_buffer_base::decimal,
            inCol0, outCol0Fnd, outCol0Exp, outCol1Fnd, outCol1Exp,
            inCol0Name,
            outCol0Name + " (found)", outCol0Name + " (expected)",
            outCol1Name + " (found)", outCol1Name + " (expected)"
    );
    
    const equality_check ec0(outCol0Exp, outCol0Fnd);
    const equality_check ec1(outCol1Exp, outCol1Fnd);
    const bool allGood = ec0.good() && ec1.good();
    
    std::cout << std::endl;
    print_check(outCol0Name, ec0);
    print_check(outCol1Name, ec1);
    print_overall(allGood);
    
    return allGood;
}

bool test_op_2in_2out_1val(
        const std::string & opName,
        op_2in_2out_1val op,
        const column<uncompr_f> * inCol0,
        const column<uncompr_f> * inCol1,
        const std::string & inCol0Name,
        const std::string & inCol1Name,
        const column<uncompr_f> * outCol0Exp,
        const column<uncompr_f> * outCol1Exp,
        const std::string & outCol0Name,
        const std::string & outCol1Name,
        size_t val0
) {
    print_header(opName);
    
    const column<uncompr_f> * outCol0Fnd;
    const column<uncompr_f> * outCol1Fnd;
    std::tie(outCol0Fnd, outCol1Fnd) = op(inCol0, inCol1, val0);
    
    print_columns(
            print_buffer_base::decimal,
            inCol0, inCol1, outCol0Fnd, outCol0Exp, outCol1Fnd, outCol1Exp,
            inCol0Name, inCol1Name,
            outCol0Name + " (found)", outCol0Name + " (expected)",
            outCol1Name + " (found)", outCol1Name + " (expected)"
    );
    
    const equality_check ec0(outCol0Exp, outCol0Fnd);
    const equality_check ec1(outCol1Exp, outCol1Fnd);
    const bool allGood = ec0.good() && ec1.good();
    
    std::cout << std::endl;
    print_check(outCol0Name, ec0);
    print_check(outCol1Name, ec1);
    print_overall(allGood);
    
    return allGood;
}

}
#endif //MORPHSTORE_CORE_OPERATORS_OPERATOR_TEST_FRAMES_H
