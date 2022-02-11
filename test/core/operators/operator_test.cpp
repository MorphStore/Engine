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

#include <vector>

/// basics
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>

/// storage
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

/// utils
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>
#include <core/utils/equality_check.h>

/// operators
#include <core/operators/interfaces/select.h>
#include <core/operators/uncompr/select.h>
#include <core/operators/reference/select.h>
#include <core/operators/virtual_vectorized/select_uncompr.h>

#include <core/operators/interfaces/project.h>
#include <core/operators/uncompr/project.h>
#include <core/operators/reference/project.h>
#include <core/operators/virtual_vectorized/project_uncompr.h>

#include <core/operators/interfaces/group_first.h>
#include <core/operators/uncompr/group_first.h>
#include <core/utils/string_manipulation.h>

using namespace morphstore;
using namespace vectorlib;


/// function types
using select_operator_t = const column<uncompr_f> * (*)(const column<uncompr_f> *, const uint64_t, const size_t);
using project_operator_t = const column<uncompr_f> * (*)(const column<uncompr_f> *, const column<uncompr_f> *);

/// comparator for select operator
template<class VectorExtension, int Granularity>
using comparator = typename vectorlib::less<VectorExtension, Granularity>;


template< template<class, int> class comparator >
int test_select(
  const column<uncompr_f> * const inColumn1,
  const std::vector<std::pair<std::string, select_operator_t>> * selectOperators,
  const int64_t comparatorValue
) {
    using scalar_comparator = comparator<scalar<v64<uint64_t>>, 64>;
    /// first: check if reference implementation works correct
    /// create test data
    std::vector<uint64_t> inData {0, 3, 7, 13, 4, 27, 44, 1};
    std::vector<uint64_t> outData;
    
    /// calculate output data for select operator using given comparator
    for(uint64_t index = 0; index < inData.size(); ++index){
        if(scalar_comparator::apply(inData[index], 10))
            outData.push_back(index);
    }
    
    const column<uncompr_f> * refInCol  = ColumnGenerator::make_column(inData);
    const column<uncompr_f> * refOutCol = ColumnGenerator::make_column(outData);
    
//    info("IN column");
//    print_columns(print_buffer_base::decimal, refInCol);
//    info("OUT column");
//    print_columns(print_buffer_base::decimal, refOutCol);
    
    /// calculate reference select
    select_operator_t reference_select  = select<scalar<v64<uint64_t>>,comparator,uncompr_f,uncompr_f>;
    const column<uncompr_f> * refResult = reference_select(refInCol, 10, 0);
    
    /// compare results
    const equality_check eq0(refOutCol, refResult);
    if(eq0.good()){
        info("Reference select implementation works fine.");
    } else {
        error("Reference select implementation is corrupt. Abort!");
        return 1;
    }
    
    /// cleanup
    delete refInCol;
    delete refOutCol;
    delete refResult;
    
    /// calculate reference result using input column
    refOutCol = reference_select(inColumn1, comparatorValue, 0);
    if(refOutCol->get_count_values() <= 0){
        error("Reference column does not contain any data. Abort!");
        return 1;
    }
    
//    info("Reference column");
//    print_columns(print_buffer_base::decimal, refOutCol);
    
    bool corruptOperatorFound = false;
    
    /// test all given select impementations
    for(const auto& pair : *selectOperators){
        std::string operatorName = pair.first;
        select_operator_t operatorFnc = pair.second;
        
        /// skip empty function pointer
        if(operatorFnc == nullptr){
             warn("Test operator select with ", str_rfill(operatorName, 25, "."), "...Skipped. (received nullptr / vector extension not available?)");
             continue;
        }
        
        /// execute operator
        const column<uncompr_f> * testOutCol = operatorFnc(inColumn1, comparatorValue, 0);
        
//        info("Test column");
//        print_columns(print_buffer_base::decimal, testOutCol);
        
        /// compare results
        const equality_check eq1(refOutCol, testOutCol);
        if(eq1.good()){
            info("Test operator select with ", str_rfill(operatorName, 25, "."), "...Successful.");
        } else {
            error("Test operator select with ", str_rfill(operatorName, 25, "."), "...Failed.");
            corruptOperatorFound = true;
        }
    }
    
    return corruptOperatorFound;
}


int test_project(
  const column<uncompr_f> * const inColumn1,
  const column<uncompr_f> * const inColumn2,
  const std::vector<std::pair<std::string, project_operator_t>> * selectOperators
){

    /// first: check if reference implementation works correct
    /// create test data
    std::vector<uint64_t> inData  {0, 3, 7, 13, 4, 27, 44, 1};
    std::vector<uint64_t> inPos   {   1,     3, 4,      6};
    std::vector<uint64_t> outData {   3,    13, 4,     44};
    
    const column<uncompr_f> * refInDataCol = ColumnGenerator::make_column(inData);
    const column<uncompr_f> * refInPosCol  = ColumnGenerator::make_column(inPos);
    const column<uncompr_f> * refOutCol    = ColumnGenerator::make_column(outData);
    
    /// calculate reference select
    project_operator_t reference_project = project<scalar<v64<uint64_t>>,uncompr_f,uncompr_f,uncompr_f>;
    const column<uncompr_f> * refResult  = reference_project(refInDataCol, refInPosCol);
    
    /// compare results
    const equality_check eq0(refOutCol, refResult);
    if(eq0.good()){
        info("Reference project implementation works fine.");
    } else {
        error("Reference project implementation is corrupt. Abort!");
        return 1;
    }
    
    /// cleanup
    delete refInDataCol;
    delete refInPosCol;
    delete refOutCol;
    delete refResult;
    
    /// calculate reference result using input column
    refOutCol = reference_project(inColumn1, inColumn2);
    if(refOutCol->get_count_values() <= 0){
        error("Reference column does not contain any data. Abort!");
        return 1;
    }
    
    bool corruptOperatorFound = false;
    
    /// test all given select impementations
    for(const auto& pair : *selectOperators){
        std::string operatorName = pair.first;
        project_operator_t operatorFnc = pair.second;
        
        /// skip empty function pointer
        if(operatorFnc == nullptr){
             warn("Test operator project with ", str_rfill(operatorName, 25, "."), "...Skipped. (received nullptr / vector extension not available?)");
             continue;
        }
        
        /// execute operator
        const column<uncompr_f> * testOutCol = operatorFnc(inColumn1, inColumn2);
        
//        info("Test column");
//        print_columns(print_buffer_base::decimal, testOutCol);
        
        /// compare results
        const equality_check eq1(refOutCol, testOutCol);
        if(eq1.good()){
            info("Test operator project with ", str_rfill(operatorName, 25, "."), "...Successful.");
        } else {
            error("Test operator project with ", str_rfill(operatorName, 25, "."), "...Failed.");
            corruptOperatorFound = true;
        }
    }
    
    return corruptOperatorFound;
}




int main( void ) {
    
    while(1){
        column<uncompr_f> * col = new column<uncompr_f>(1024 * 1024 * 1024);
        delete col;
    }
    return 0;
    
    uint64_t * foo;
    uint64_t * bar = new uint64_t[10];
    {
        voidptr_t vptr = bar;
        foo = vptr;
        for (uint64_t i = 0; i < 10; ++ i) {
            foo[i] = i * 5;
        }
    }
    for(uint64_t i = 0; i < 10; ++i){
        std::cout << "(" << i << " : " << foo[i] << "),";
    }
    std::cout << std::endl;
    delete[] bar;
    
    for(uint64_t i = 0; i < 10; ++i){
        std::cout << "(" << i << " : " << foo[i] << "),";
    }
    std::cout << std::endl;
    return 0;
    
//    const column<uncompr_f> * test = make_column({0,1,2,3,4});
//    uint64_t * data = test->get_data();
//    info("1");
//    uint64_t * bla = test->get_data();
//
//    info("2");
//    delete bla;
//    info("3");
//    delete test;
//    info("4");
//    for(uint64_t i = 0; i < 5; ++i){
//        std::cout << "(" << i << " : " << data[i] << "),";
//    }
//    std::cout << std::endl;
//    return 0;
    
    
    /// ===== test data =============================================================================================///
    /// element count = multiplicator * elementsPerVector + vectorOverhead
    const size_t columnSize = 100 * 8 + 7;
    uint64_t checksum = 0;
    
    const column<uncompr_f> * const baseCol1
      = ColumnGenerator::generate_with_distr(
        columnSize,
        std::uniform_int_distribution<uint64_t>(0, 1),
        false,
        std::time(nullptr)
      );
    
    
    /// ===== select ================================================================================================///
    /// list of all test cases
    std::vector<std::pair<std::string, select_operator_t>> select_operators {
      {"Scalar<64>", select<scalar<v64<uint64_t>>, comparator, uncompr_f, uncompr_f>},
      {"vv<1024, Scalar<64>>", select<vv< v1024<uint64_t>, scalar<v64<uint64_t>> >, comparator, uncompr_f, uncompr_f>},
      
      #ifdef SSE
      {"SSE<128>", select<sse<v128<uint64_t>>, comparator, uncompr_f, uncompr_f>},
      {"vv<1024, SSE<128>>", select<vv< v1024<uint64_t>, sse<v128<uint64_t>> >, comparator, uncompr_f, uncompr_f>},
      #else
      {"SSE<128>", nullptr}
      #endif
      
      #ifdef AVXTWO
      {"AVX2<256>", select<avx2<v256<uint64_t>>, comparator, uncompr_f, uncompr_f>},
      {"vv<1024, AVX2<256>>", select<vv< v1024<uint64_t>, avx2<v256<uint64_t>> >, comparator, uncompr_f, uncompr_f>},
      #else
      {"AVX2<256>", nullptr},
      #endif
     
      #ifdef AVX512
      {"AVX512<512>", select<avx512<v512<uint64_t>>, comparator, uncompr_f, uncompr_f>},
      {"vv<1024, AVX512<512>>", select<vv< v1024<uint64_t>, avx512<v512<uint64_t>> >, comparator, uncompr_f, uncompr_f>}
      #else
      {"AVX512<512>", nullptr}
      #endif
    };
    checksum += test_select<comparator>(baseCol1, &select_operators, 1);
    
    
    /// ===== project ===============================================================================================///
    /// list of all test cases
    std::vector<std::pair<std::string, project_operator_t>> project_operators {
      {"Scalar<64>", project<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f>},
      {"vv<1024, Scalar<64>>", project<vv_old<v1024<uint64_t>, scalar<v64<uint64_t>> >, uncompr_f, uncompr_f, uncompr_f>},
      
      #ifdef SSE
      {"SSE<128>", project<sse<v128<uint64_t>>, uncompr_f, uncompr_f, uncompr_f>},
      {"vv<1024, SSE<128>>", project<vv_old<v1024<uint64_t>, sse<v128<uint64_t>> >, uncompr_f, uncompr_f, uncompr_f>},
      #else
      {"SSE<128>", nullptr}
      #endif
      
      #ifdef AVXTWO
      {"AVX2<256>", project<avx2<v256<uint64_t>>, uncompr_f, uncompr_f, uncompr_f>},
      {"vv<1024, AVX2<256>>", project<vv_old<v1024<uint64_t>, avx2<v256<uint64_t>> >, uncompr_f, uncompr_f, uncompr_f>},
      #else
      {"AVX2<256>", nullptr},
      #endif
     
      #ifdef AVX512
      {"AVX512<512>", project<avx512<v512<uint64_t>>, uncompr_f, uncompr_f, uncompr_f>},
      {"vv<1024, AVX512<512>>", project<vv_old<v1024<uint64_t>, avx512<v512<uint64_t>> >, uncompr_f, uncompr_f, uncompr_f>}
      #else
      {"AVX512<512>", nullptr}
      #endif
    };
    
    /// generate position column for baseCol1 (assuming the reference implementation works correct)
    const column<uncompr_f> * positionColumn
      = select<scalar<v64<uint64_t>>, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 1);
    
    checksum += test_project(baseCol1, positionColumn, &project_operators);
    
    
    
    if(checksum){
        error("There are some corrupted operators among us. Hide!");
    } else {
        info("All tested operators work fine. Nice!");
    }
}
