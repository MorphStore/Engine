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
 * @file static_vbp_test.cpp
 * @brief Tests of the (de)compression morph operators for the static_vbp
 * format.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/equality_check.h>
#include <core/utils/monitoring.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/simd/sse/extension_sse.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace morphstore;
using namespace vector;

template< unsigned bw >
bool test( ) {
    MONITORING_CREATE_MONITOR( MONITORING_MAKE_MONITOR(bw), MONITORING_KEY_IDENTS("bitwidth"));
    
    MONITORING_START_INTERVAL_FOR( "operatorTime", bw );
    
    // Generate some data.
    const uint64_t minVal = (bw == 1)
            ? 0
            : (static_cast<uint64_t>(1) << (bw - 1));
    const uint64_t maxVal = (bw == 64)
            ? std::numeric_limits<uint64_t>::max()
            : ((static_cast<uint64_t>(1) << bw) - 1);
    auto origCol = generate_with_distr(
            128 * 1024,
            std::uniform_int_distribution<uint64_t>(minVal, maxVal),
            false
    );

    // Compress the data (using the scalar and the vec128 morph-operator).
    auto comprColScalar = morph<
            scalar<v64<uint64_t>>,
            static_vbp_f<bw, sizeof(__m128i) / sizeof(uint64_t)>
    >(origCol);
    auto comprColVec128 = morph<
            sse<v128<uint64_t>>,
            static_vbp_f<bw, sizeof(__m128i) / sizeof(uint64_t)>
    >(origCol);
    
    // Comparison of the compressed columns.
    const bool equalCompr = equality_check(
            comprColScalar, comprColVec128
    ).good();
    
    // Decompress the data (using the scalar and the vec128 morph-operator).
    auto decomprColScalar = morph<scalar<v64<uint64_t>>, uncompr_f>(
            comprColScalar
    );
    auto decomprColVec128 = morph<sse<v128<uint64_t>>, uncompr_f>(
            comprColVec128
    );
    
    // Comparison of the decompressed columns.
    const bool equalDecompr = equality_check(
            decomprColScalar, decomprColVec128
    ).good();
    
    MONITORING_END_INTERVAL_FOR("operatorTime", bw);
    MONITORING_START_INTERVAL_FOR("operatorTime", bw);
    MONITORING_END_INTERVAL_FOR("operatorTime", bw);

    // Comparison of the decompressed and the original data.
    const bool goodDecompr = equality_check(origCol, decomprColScalar).good();
    
    // Overall check.
    const bool allGood = equalCompr && equalDecompr && goodDecompr;
    std::cout
            << std::setw(2) << bw << " bit: "
            << equality_check::ok_str(allGood) << std::endl;
    MONITORING_ADD_BOOL_FOR("ok", allGood, bw);
    MONITORING_ADD_BOOL_FOR("ok", allGood, bw);

    return allGood;
}

int main( void ) {
    bool allGood = true;
#if 0
    for( unsigned bw = 1; bw <= 64; bw++ )
        allGood = allGood && test< bw >( );
#else
    // Generated with Python:
    // for bw in range(1, 64+1):
    //   print("allGood = allGood && test<{: >2}>();".format(bw))
    allGood = allGood && test< 1>();
    allGood = allGood && test< 2>();
    allGood = allGood && test< 3>();
    allGood = allGood && test< 4>();
    allGood = allGood && test< 5>();
    allGood = allGood && test< 6>();
    allGood = allGood && test< 7>();
    allGood = allGood && test< 8>();
    allGood = allGood && test< 9>();
    allGood = allGood && test<10>();
    allGood = allGood && test<11>();
    allGood = allGood && test<12>();
    allGood = allGood && test<13>();
    allGood = allGood && test<14>();
    allGood = allGood && test<15>();
    allGood = allGood && test<16>();
    allGood = allGood && test<17>();
    allGood = allGood && test<18>();
    allGood = allGood && test<19>();
    allGood = allGood && test<20>();
    allGood = allGood && test<21>();
    allGood = allGood && test<22>();
    allGood = allGood && test<23>();
    allGood = allGood && test<24>();
    allGood = allGood && test<25>();
    allGood = allGood && test<26>();
    allGood = allGood && test<27>();
    allGood = allGood && test<28>();
    allGood = allGood && test<29>();
    allGood = allGood && test<30>();
    allGood = allGood && test<31>();
    allGood = allGood && test<32>();
    allGood = allGood && test<33>();
    allGood = allGood && test<34>();
    allGood = allGood && test<35>();
    allGood = allGood && test<36>();
    allGood = allGood && test<37>();
    allGood = allGood && test<38>();
    allGood = allGood && test<39>();
    allGood = allGood && test<40>();
    allGood = allGood && test<41>();
    allGood = allGood && test<42>();
    allGood = allGood && test<43>();
    allGood = allGood && test<44>();
    allGood = allGood && test<45>();
    allGood = allGood && test<46>();
    allGood = allGood && test<47>();
    allGood = allGood && test<48>();
    allGood = allGood && test<49>();
    allGood = allGood && test<50>();
    allGood = allGood && test<51>();
    allGood = allGood && test<52>();
    allGood = allGood && test<53>();
    allGood = allGood && test<54>();
    allGood = allGood && test<55>();
    allGood = allGood && test<56>();
    allGood = allGood && test<57>();
    allGood = allGood && test<58>();
    allGood = allGood && test<59>();
    allGood = allGood && test<60>();
    allGood = allGood && test<61>();
    allGood = allGood && test<62>();
    allGood = allGood && test<63>();
    allGood = allGood && test<64>();
#endif

    MONITORING_PRINT_MONITORS(monitorCsvLog);
	
    std::cout << "overall: " << equality_check::ok_str(allGood) << std::endl;

#ifdef MSV_USE_MONITORING
    std::cout << "Monitoring was ENABLED" << std::endl;
#else
    std::cout << "Monitoring was DISABLED" << std::endl;
#endif
    
    return !allGood;
}
