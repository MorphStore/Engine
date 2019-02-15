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
 * @file binary_io_test.cpp
 * @brief A short test and example usage of persistence::binary_io .
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/persistence/binary_io.h"
#include "../../../include/core/storage/column.h"
#include "../../../include/core/utils/equality_check.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

using namespace morphstore;

int main( void ) {
    // Parameters.
    const size_t origCountValues = 100 * 1000;
    const size_t origSizeUsedByte = origCountValues * sizeof( uint64_t );
    const std::string fileName = "binary_io_test__testcol123";
    
    // Create the column.
    auto origCol = new column< uncompr_f >( origSizeUsedByte );
    uint64_t * origData = origCol->get_data( );
    for( unsigned i = 0; i < origCountValues; i++ )
        origData[ i ] = i;
    origCol->set_count_values( origCountValues );
    origCol->set_size_used_byte( origSizeUsedByte );
    
    // Store the column.
    // TODO maybe we should delete the file afterwards
    binary_io< uncompr_f >::store( origCol, fileName );
    
    // Reload the column and compare it to the original one.
    auto reloCol = binary_io< uncompr_f >::load( fileName );
    
    // Compare the original column to the reloaded column.
    std::cout << equality_check( origCol, reloCol );
    
    return 0;
}