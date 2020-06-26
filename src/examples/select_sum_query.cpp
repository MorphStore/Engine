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
//#include <core/operators/general_vectorized/calc_uncompr.h>
// #include <core/operators/general_vectorized/select_compr.h>
// #include <core/operators/general_vectorized/select_uncompr.h>
#include <core/operators/specialized/select_type_packing.h>
#include <core/operators/specialized/select_type_packing_32to32.h>
#include <core/operators/specialized/select_type_packing_32to64.h>
#include <core/operators/specialized/select_type_packing_64to32.h>
#include <core/operators/specialized/calc_unary_type_packing.h>
#include <core/operators/specialized/calc_unary_type_packing_16to32.h>
#include <core/operators/specialized/calc_unary_type_packing_32to8.h>


#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>
#include <core/utils/math.h>

#include <core/morphing/static_vbp.h>
#include <core/morphing/vbp.h>
#include <core/morphing/morph.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <vector/type_helper.h>

#include <functional>
#include <iostream>
#include <random>

#include <core/morphing/type_packing.h>
#include <core/morphing/type_packing_vertical.h>

using namespace morphstore;
using namespace vectorlib;

// ****************************************************************************
// * Example query
// ****************************************************************************


int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    const size_t countValues = 16;//189;//100 * 1000 * 1000;
    const column<uncompr_f> * const baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(100, 199),
            false,
            2
    );

    // const column<uncompr_f> * const baseCol2 = generate_with_distr(
    //         countValues,
    //         std::uniform_int_distribution<uint64_t>(0, 10),
    //         false,
    //         2
    // );

    print_columns(print_buffer_base::decimal, baseCol1, "SUM(baseCol1)");

    //using ve = scalar<v64<uint64_t> >;
    //using ve = sse<v128<uint64_t> >;
    using ve = avx2<v256<uint64_t> >;
    //using ve= avx512<v512<uint64_t> >;
    //using ve = TypeHelper<ve1, uint64_t>::newbasetype;

    //auto baseCol1Compr = morph<ve, static_vbp_f<vbp_l<64, 1> > >(baseCol1);
    //auto baseCol2Compr = morph<ve, static_vbp_f<vbp_l<4, 1> > >(baseCol2);
    //auto baseCol1Compr = morph<ve, type_packing_f<uint32_t > >(baseCol1);
    //auto baseCol2Compr = morph<ve, type_packing_f<uint32_t > >(baseCol1);
    auto baseCol1Compr = morph<ve, type_packing_f<uint16_t > >(baseCol1);
    //print_columns(print_buffer_base::hexadecimal, baseCol1Compr, "SUM(baseCol1)");
    //auto baseCol1Decompr = morph<ve, uncompr_f >(baseCol1Compr);
    //print_columns(print_buffer_base::hexadecimal, baseCol1Decompr, "baseCol1Decompr");

    std::cout << "done." << std::endl;
    
    // ************************************************************************
    // * Query execution
    // ************************************************************************
    
    std::cout << "Query execution started... ";
    std::cout.flush();

    auto i1 = calc_unary_t<
            add,
            ve,
            type_packing_f<uint32_t > , //output format of operator
            type_packing_f<uint16_t >   //input format of operator
    >::apply(baseCol1Compr, 5);


    // auto i1 = my_select_wit_t<
    //         equal,
    //         ve,
    //         static_vbp_f<vbp_l<27, 1> >, //output format of operator
    //         static_vbp_f<vbp_l<8, 1> >  //input format of operator
    // >::apply(baseCol1Compr, 150);

    // auto i1 = my_select_wit_t<
    //         equal,
    //         ve,
    //         type_packing_f<uint32_t > , //output format of operator
    //         type_packing_f<uint32_t >   //input format of operator
    // >::apply(baseCol1Compr, 150);

    // auto i1 = select_t<
    //         equal,
    //         ve,
    //         type_packing_f<uint32_t>, //output format of operator
    //         type_packing_f<uint64_t>  //input format of operator
    // >::apply(baseCol1Compr, 150);

    auto i2 = morph<ve, uncompr_f >(i1);

    // auto i1 = morphstore::select<
    //         equal,
    //         ve,
    //         uncompr_f, //output format of operator
    //         uncompr_f  //input format of operator
    // >(baseCol1, 150);    

    // auto i1 = my_select_wit_t<
    //         equal,
    //         ve,
    //         uncompr_f,
    //         //type_packing_f<uint64_t >,
    //         type_packing_f<uint8_t >
    // >::apply(baseCol1Compr, 150);

    //auto i2 = morph<ve, uncompr_f >(i1);
    // auto i2 = my_project_wit_t<
    //         ve,
    //         static_vbp_f<vbp_l<4, 1> >, //output format of operator
    //         static_vbp_f<vbp_l<4, 1> >, //input 1 format of operator
    //         static_vbp_f<vbp_l<27, 1> >  //input 2 format of operator
    // >::apply(baseCol2Compr, i1);

    // auto i2 = my_project_wit_t<
    //         ve,
    //         type_packing_f<uint64_t >, //output format of operator           
    //         uncompr_f, //input 1 format of operator
    //         uncompr_f //input 2 format of operator
    // >::apply(baseCol2, i1);


    // auto i3 = agg_sum<
    //         ve, 
    //         static_vbp_f<morphstore::vbp_l<4, 1> > //input format of operator
    // >(i2);

    // auto i3 = agg_sum<
    //         ve, 
    //         type_packing_f<uint64_t > //input format of operator
    // >(i2);    

    std::cout << "done." << std::endl << std::endl;
    
    // // ************************************************************************
    // // * Result output
    // // ************************************************************************
    print_columns(print_buffer_base::decimal, i1, "SUM(baseCol2)");
    //print_columns(print_buffer_base::hexadecimal, i1compr, "i1compr)");

    print_columns(print_buffer_base::decimal, i2, "SUM(baseCol2)");
    return 0;
}




// #include <core/memory/mm_glob.h>
// #include <core/morphing/format.h>
// #include <core/morphing/uncompr.h>
// #include <core/operators/general_vectorized/agg_sum_compr.h>
// #include <core/operators/general_vectorized/project_compr.h>
// #include <core/operators/general_vectorized/select_compr.h>
// #include <core/storage/column.h>
// #include <core/storage/column_gen.h>
// #include <core/utils/basic_types.h>
// #include <core/utils/printing.h>
// #include <core/utils/math.h>

// #include <vector/vector_extension_structs.h>
// #include <vector/vector_primitives.h>

// #include <functional>
// #include <iostream>
// #include <random>

// using namespace morphstore;
// using namespace vectorlib;

// // ****************************************************************************
// // * Example query
// // ****************************************************************************

// // SELECT SUM(baseCol2) WHERE baseCol1 = 150

// int main( void ) {
//     // ************************************************************************
//     // * Generation of the synthetic base data
//     // ************************************************************************
    
//     std::cout << "Base data generation started... ";
//     std::cout.flush();
    
//     const size_t countValues = 189; // 100 * 1000 * 1000;
//     const column<uncompr_f> * const baseCol1 = generate_with_distr(
//             countValues,
//             std::uniform_int_distribution<uint64_t>(100, 199),
//             false,
//             2
//     );
//     const column<uncompr_f> * const baseCol2 = generate_with_distr(
//             countValues,
//             std::uniform_int_distribution<uint64_t>(0, 10),
//             false,
//             2
//     );
    
//     std::cout << "done." << std::endl;
    
//     // ************************************************************************
//     // * Query execution
//     // ************************************************************************
    
//     using ve = scalar<v64<uint64_t> >;
    
//     std::cout << "Query execution started... ";
//     std::cout.flush();
    
//     // Positions fulfilling "baseCol1 = 150"
//     auto i1 = my_select_wit_t<
//             equal,
//             ve,
//             uncompr_f,
//             uncompr_f
//     >::apply(baseCol1, 150);

//     // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
//     auto i2 = my_project_wit_t<
//             ve,
//             uncompr_f,
//             uncompr_f,
//             uncompr_f
//     >::apply(baseCol2, i1);
//     // Sum over the data elements of "baseCol2" fulfilling "baseCol1 = 150"
//     auto i3 = agg_sum<ve, uncompr_f>(i2);
    

//     std::cout << "done." << std::endl << std::endl;
    
//     // ************************************************************************
//     // * Result output
//     // ************************************************************************

//     print_columns(print_buffer_base::decimal, i3, "SUM(baseCol2)");
    
//     return 0;
// }



// #include "../../include/core/memory/mm_glob.h"
// #include "../../include/core/morphing/format.h"
// #include "../../include/core/operators/scalar/agg_sum_uncompr.h"
// #include "../../include/core/operators/scalar/project_uncompr.h"
// #include "../../include/core/operators/scalar/select_uncompr.h"
// #include "../../include/core/storage/column.h"
// #include "../../include/core/storage/column_gen.h"
// #include "../../include/core/utils/basic_types.h"
// #include "../../include/core/utils/printing.h"
// #include "../../include/vector/scalar/extension_scalar.h"

// #include <functional>
// #include <iostream>
// #include <random>

// using namespace morphstore;
// using namespace vectorlib;

// // ****************************************************************************
// // * Example query
// // ****************************************************************************

// // SELECT SUM(baseCol2) WHERE baseCol1 = 150

// int main( void ) {
//     // ************************************************************************
//     // * Generation of the synthetic base data
//     // ************************************************************************
    
//     std::cout << "Base data generation started... ";
//     std::cout.flush();
    
//     const size_t countValues = 189;
//     const column<uncompr_f> * const baseCol1 = generate_with_distr(
//             countValues,
//             std::uniform_int_distribution<uint64_t>(100, 199),
//             false,
//             2
//     );
//     // const column<uncompr_f> * const baseCol2 = generate_with_distr(
//     //         countValues,
//     //         std::uniform_int_distribution<uint64_t>(0, 10),
//     //         false
//     // );
    
//     std::cout << "done." << std::endl;
    
//     // ************************************************************************
//     // * Query execution
//     // ************************************************************************
    
//     using ve = scalar<v64<uint64_t> >;
    
//     std::cout << "Query execution started... ";
//     std::cout.flush();
    
//     // Positions fulfilling "baseCol1 = 150"
//     auto i1 = morphstore::select<
//             std::equal_to,
//             ve,
//             uncompr_f,
//             uncompr_f
//     >(baseCol1, 150);
//     // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
//     //auto i2 = project<ve, uncompr_f>(baseCol2, i1);
//     // Sum over the data elements of "baseCol2" fulfilling "baseCol1 = 150"
//     //auto i3 = agg_sum<ve, uncompr_f>(i2);
    
//     std::cout << "done." << std::endl << std::endl;
    
//     // ************************************************************************
//     // * Result output
//     // ************************************************************************

//     print_columns(print_buffer_base::decimal, i1, "SUM(baseCol2)");
    
//     return 0;
// }