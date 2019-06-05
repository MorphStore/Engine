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
 * @file printing_test.cpp
 * @brief A short test and example usage of print_columns .
 * @todo Move this file to some examples directory, since it is no actual test.
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/printing.h>
#include <vector/simd/sse/extension_sse.h>

#include <cstdint>
#include <immintrin.h>
#include <unordered_map>

using namespace morphstore;
using namespace vector;

/**
 * A small example usage of print_columns.
 */
void small_example( ) {
    auto origCol = generate_sorted_unique(128);
    auto comprCol = morph<
            sse<v128<uint64_t>>,
            static_vbp_f<8, sizeof(__m128i) / sizeof(uint64_t)>
    >(origCol);
    auto decomprCol = morph<sse<v128<uint64_t>>, uncompr_f>(comprCol);
    
    print_columns(
            print_buffer_base::hexadecimal,
            origCol,
            comprCol,
            decomprCol,
            "original",
            "compressed",
            "decompressed"
    );
}

template< typename uintX_t >
void systematic_test_internal(
        const column< uncompr_f > * col1,
        const column< uncompr_f > * col2,
        const column< uncompr_f > * col3
) {
    std::unordered_map< print_buffer_base, std::string > map = {
        { print_buffer_base::binary     , "binary"      },
        { print_buffer_base::decimal    , "decimal"     },
        { print_buffer_base::hexadecimal, "hexadecimal" },
    };

    for( print_buffer_base pbb : {
        print_buffer_base::binary,
        print_buffer_base::decimal,
        print_buffer_base::hexadecimal
    } ) {
        std::cout
                << '(' << std::numeric_limits< uintX_t >::digits << " bits per line, "
                << map[pbb] << ')' << std::endl;
        print_columns< uintX_t >(
                pbb,
                col1,
                col2,
                col3,
                "very very looooooong column name",
                "c2"
        );
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/**
 * A systematic test of all bases and all element widths, showing several
 * special situations, e.g., buffer sizes not divisible by the element width
 * etc..
 */
void systematic_test( ) {
    // @todo use columngen here
    
    const size_t countValues1 = 10;
    const size_t sizeByte1 = countValues1 * sizeof( uint64_t );
    auto col1 = new column< uncompr_f >( sizeByte1 );
    uint64_t * const col1Data = col1->get_data();
    for( unsigned i = 0; i < countValues1; i++ )
        col1Data[ i ] = i;
    col1->set_count_values( countValues1 );
    col1->set_size_used_byte( sizeByte1 );
    
    const size_t countValues2 = 5;
    const size_t sizeByte2 = countValues2 * sizeof( uint64_t );
    auto col2 = new column< uncompr_f >( sizeByte2 );
    uint64_t * const col2Data = col2->get_data();
    for( unsigned i = 0; i < countValues2; i++ )
        col2Data[ i ] = std::numeric_limits< uint64_t >::max( );
    col2->set_count_values( countValues2 );
    col2->set_size_used_byte( sizeByte2 );
    
    const size_t countValues3 = 10;
    const size_t sizeByte3 = countValues3 * sizeof( uint64_t );
    auto col3 = new column< uncompr_f >( sizeByte3 );
    uint64_t * const col3Data = col3->get_data();
    for( unsigned i = 0; i < countValues3; i++ )
        col3Data[ i ] = i * 111;
    col3->set_count_values( countValues3 );
    // -5 to show the case of bytes beyond the buffer's end.
    col3->set_size_used_byte( sizeByte3 - 5 );
    
    systematic_test_internal< uint8_t >( col1, col2, col3 );
    systematic_test_internal< uint16_t >( col1, col2, col3 );
    systematic_test_internal< uint32_t >( col1, col2, col3 );
    systematic_test_internal< uint64_t >( col1, col2, col3 );
}

int main( void ) {
    small_example( );
//    systematic_test( );
    
    return 0;
}
