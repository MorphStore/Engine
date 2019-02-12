/**********************************************************************************************
 * Copyright (C) 2019 by Patrick Damme                                                        *
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
 * @author Patrick Damme
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/morphing/format.h"
#include "../../../include/core/morphing/static_vbp.h"
#include "../../../include/core/storage/column.h"
#include "../../../include/core/utils/printing.h"

#include <cstdint>
#include <unordered_map>
#include <c++/7/bits/unordered_map.h>

namespace ms = morphstore;
namespace m = morphstore::morphing;
namespace s = morphstore::storage;

/**
 * A small example usage of print_columns.
 */
void small_example( ) {
    const size_t origCountValues = 128;
    const size_t origSizeByte = origCountValues * sizeof( uint64_t );
    auto origCol = new s::column< m::uncompr_f >( origSizeByte );
    uint64_t * const origData = origCol->data();
    for( unsigned i = 0; i < origCountValues; i++ )
        origData[ i ] = i;
    origCol->count_values( origCountValues );
    origCol->size_used_byte( origSizeByte );
    
    auto comprCol = new s::column< m::static_vbp_f< 8 > >( origSizeByte );
    m::morph( origCol, comprCol );
    
    auto decomprCol = new s::column< m::uncompr_f >( origSizeByte );
    m::morph( comprCol, decomprCol );
    
    morphstore::print_columns(
            morphstore::print_buffer_base::hexadecimal,
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
        const morphstore::storage::column< morphstore::morphing::uncompr_f > * col1,
        const morphstore::storage::column< morphstore::morphing::uncompr_f > * col2,
        const morphstore::storage::column< morphstore::morphing::uncompr_f > * col3
) {
    std::unordered_map< ms::print_buffer_base, std::string > map = {
        { ms::print_buffer_base::binary     , "binary"      },
        { ms::print_buffer_base::decimal    , "decimal"     },
        { ms::print_buffer_base::hexadecimal, "hexadecimal" },
    };

    for( ms::print_buffer_base pbb : {
        ms::print_buffer_base::binary,
        ms::print_buffer_base::decimal,
        ms::print_buffer_base::hexadecimal
    } ) {
        std::cout
                << '(' << std::numeric_limits< uintX_t >::digits << " bits per line, "
                << map[pbb] << ')' << std::endl;
        ms::print_columns< uintX_t >(
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
    const size_t countValues1 = 10;
    const size_t sizeByte1 = countValues1 * sizeof( uint64_t );
    auto col1 = new s::column< m::uncompr_f >( sizeByte1 );
    uint64_t * const col1Data = col1->data();
    for( unsigned i = 0; i < countValues1; i++ )
        col1Data[ i ] = i;
    col1->count_values( countValues1 );
    col1->size_used_byte( sizeByte1 );
    
    const size_t countValues2 = 5;
    const size_t sizeByte2 = countValues2 * sizeof( uint64_t );
    auto col2 = new s::column< m::uncompr_f >( sizeByte2 );
    uint64_t * const col2Data = col2->data();
    for( unsigned i = 0; i < countValues2; i++ )
        col2Data[ i ] = std::numeric_limits< uint64_t >::max( );
    col2->count_values( countValues2 );
    col2->size_used_byte( sizeByte2 );
    
    const size_t countValues3 = 10;
    const size_t sizeByte3 = countValues3 * sizeof( uint64_t );
    auto col3 = new s::column< m::uncompr_f >( sizeByte3 );
    uint64_t * const col3Data = col3->data();
    for( unsigned i = 0; i < countValues3; i++ )
        col3Data[ i ] = i * 111;
    col3->count_values( countValues3 );
    // -5 to show the case of bytes beyond the buffer's end.
    col3->size_used_byte( sizeByte3 - 5 );
    
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