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
 * @file dynamic_vbp_test.cpp
 * @brief Tests of the (de)compression morph operators for the dynamic_vbp
 * format.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/equality_check.h>
#include <core/utils/math.h>
#include <vector/simd/sse/extension_sse.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <random>

#include <immintrin.h>

using namespace morphstore;
using namespace vector;

template<unsigned bw>
bool test() {
    // Generate some data.
    const uint64_t maxVal = (bw == 64)
            ? std::numeric_limits<uint64_t>::max()
            : ((static_cast<uint64_t>(1) << bw) - 1);
    auto origCol = generate_with_distr(
            128 * 1024,
            std::uniform_int_distribution<uint64_t>(0, maxVal),
            true
    );
    
    // @todo Use a size depending on the bit width.
    // This template specialization is SIMD-BP128 with the difference that we
    // use 64-bit instead of 32-bit data elements.
    auto comprCol = morph<
            sse<v128<uint64_t>>,
            dynamic_vbp_f<sizeof(__m128i) * bitsPerByte, sizeof(__m128i)>
    >(origCol);
    
    auto decomprCol = morph<sse<v128<uint64_t>>, uncompr_f>(comprCol);
    
    const bool good = equality_check(origCol, decomprCol).good();
    std::cout
            << std::setw(2) << bw << " bit: "
            << equality_check::ok_str(good) << std::endl;

    return good;
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
    
    std::cout << "overall: " << equality_check::ok_str(allGood) << std::endl;

    return !allGood;
}
