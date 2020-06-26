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
 * @file select_sum_query.cpp
 * @brief A little example query with a selection on one column and a sum on
 * another column.
 */
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/operators/general_vectorized/project_compr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>
#include <core/utils/math.h>

#include <core/morphing/k_wise_ns.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/morphing/morph.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <functional>
#include <iostream>
#include <random>

#include <core/morphing/type_packing.h>
#include <core/morphing/type_packing_vertical.h>
#include <core/utils/monitoring.h>

#include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/operators/general_vectorized/project_uncompr.h>
#include <core/operators/general_vectorized/select_uncompr.h>

#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"

using namespace morphstore;
using namespace vectorlib;

// ****************************************************************************
// * Example query
// ****************************************************************************


//int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    // MONITORING_CREATE_MONITOR(
    //     MONITORING_MAKE_MONITOR("uint32_t", "byte-perm"),
    //     {"type", "variant"});

 //    std::cout << "Base data generation started... ";
 //    std::cout.flush();
    
 //    const size_t countValues = 10;//1000000;  
 //    const column<uncompr_f> * const baseCol1 = generate_with_distr(
 //            countValues,
 //            std::uniform_int_distribution<uint64_t>(10, 199),
 //            false,
 //            3
 //    );

 //    print_columns(print_buffer_base::hexadecimal, baseCol1, "baseCol1");
 //  //  using ve = scalar<v64<uint64_t> >;
 //    //using ve = sse<v128<uint64_t> >;
 //    using ve = avx2<v256<uint64_t>>;
 //    //using ve = avx512<v512<uint64_t>>;
 //    //compression

 // //   MONITORING_START_INTERVAL_FOR("time", "uint32_t", "byte-perm");
 //    auto baseCol1Compr = morph<ve, type_packing_f<uint64_t > >(baseCol1);
 // //   MONITORING_END_INTERVAL_FOR("time", "uint32_t", "byte-perm");
 // //   MONITORING_PRINT_MONITORS(monitorCsvLog);
 //    //std::cout << "get_size_compr_byte baseCol1 " << baseCol1Compr->get_size_compr_byte() << std::endl;
 //    print_columns(print_buffer_base::hexadecimal, baseCol1Compr, "baseCol1Compr");
 //    //decompression
 //    auto baseCol1Decompr = morph<ve, uncompr_f >(baseCol1Compr);
 //    print_columns(print_buffer_base::hexadecimal, baseCol1Decompr, "baseCol1Decompr");

 //    std::cout << "done." << std::endl;
    

 //    return 0;


    // { //test add
    // using ps1 = avx2<v256<uint64_t>>;
    // IMPORT_VECTOR_BOILER_PLATE_PREFIX(ps1, ps_)

    //  ps_vector_t sequence1 = set_sequence<ps1, ps_vector_base_t_granularity::value>(0,1); //0,1,2,...,7
    //  ps_vector_t sequence2 = set1<ps1, ps_vector_base_t_granularity::value>(1);     //1,1,...,1
    //  std::cout<< "using avx2, 32bit" << std::endl;  
    //  ps_vector_t added = add<ps1, ps_vector_base_t_granularity::value>::apply(sequence1, sequence2); //1,2,...,8
    //  uint64_t number1 = 0;
    //  for(int i=0; i < 4; i++){
    //      number1 = extract_value<ps1,ps_vector_base_t_granularity::value>(added, i);
    //      std::cout << number1 << std::endl;
    //  }
    // }    

    
//}

int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    const size_t countValues = 100; 
    const column<uncompr_f> * const baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(100, 199),
            false,
            2
    );

    print_columns(print_buffer_base::hexadecimal, baseCol1, "SUM(baseCol1)");
  //  using ve = scalar<v64<uint64_t> >;
    using ve = sse<v128<uint64_t> >;
    //using ve = avx2<v256<uint64_t> >;

    //compression
    auto baseCol1Compr = morph<ve, type_packing_f<uint8_t > >(baseCol1);
    std::cout << "get_size_compr_byte baseCol1 " << baseCol1Compr->get_size_compr_byte() << std::endl;
    print_columns(print_buffer_base::hexadecimal, baseCol1Compr, "SUM(baseCol1)");
    //decompression
    auto baseCol1Decompr = morph<ve, uncompr_f >(baseCol1Compr);
    print_columns(print_buffer_base::hexadecimal, baseCol1Decompr, "baseCol1Decompr)");

    std::cout << "done." << std::endl;
    

    return 0;
}






