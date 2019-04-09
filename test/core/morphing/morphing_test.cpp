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
 * @file morphing_test.cpp
 * @brief ...
 * @todo TODOS?
 */

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/static_vbp.h>
#include <core/storage/column.h>
#include <core/utils/equality_check.h>

#include <core/utils/monitoring.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>

using namespace morphstore;

template< unsigned bw >
bool test( ) {
	MONITOR_START_INTERVAL( "operatorTime" + std::to_string( bw ) );
	const size_t origCountValues = 128 * 1024;
    const size_t origSizeByte = origCountValues * sizeof( uint64_t );
    auto origCol = new column< uncompr_f >( origSizeByte );
    uint64_t * origData = origCol->get_data( );
    const uint64_t mask = ( bw == 64 )
        ? std::numeric_limits< uint64_t >::max( )
        : ( ( static_cast< uint64_t >( 1 ) << bw ) - 1 );
    for( unsigned i = 0; i < origCountValues; i++ )
        origData[ i ] = i & mask;
    origCol->set_count_values( origCountValues );
    origCol->set_size_used_byte( origSizeByte );
    
    // TODO use a size depending on the bit width
    auto comprCol = new column< static_vbp_f< bw > >( origSizeByte );
    morph( origCol, comprCol );
    
    // TODO use a size depending on the bit width
    auto decomprCol = new column< uncompr_f >( origSizeByte );
    morph( comprCol, decomprCol );
    
    const bool good = equality_check( origCol, decomprCol ).good( );
    std::cout
            << std::setw(2) << bw << " bit: "
            << equality_check::ok_str( good ) << std::endl;
    MONITOR_END_INTERVAL( "operatorTime" + std::to_string( bw ) );

    MONITOR_ADD_PROPERTY( "operatorParam" + std::to_string( bw ), bw );
    return good;
}

int main( void ) {
    bool allGood = true;
#if 0
    for( unsigned bw = 1; bw <= 64; bw++ )
        allGood = allGood && test< bw >( );
#else
    allGood = allGood && test<  1 >( );
    allGood = allGood && test<  2 >( );
    allGood = allGood && test<  3 >( );
    allGood = allGood && test<  4 >( );
    allGood = allGood && test<  5 >( );
    allGood = allGood && test<  6 >( );
    allGood = allGood && test<  7 >( );
    allGood = allGood && test<  8 >( );
    allGood = allGood && test<  9 >( );
    allGood = allGood && test< 10 >( );
    allGood = allGood && test< 11 >( );
    allGood = allGood && test< 12 >( );
    allGood = allGood && test< 13 >( );
    allGood = allGood && test< 14 >( );
    allGood = allGood && test< 15 >( );
    allGood = allGood && test< 16 >( );
    allGood = allGood && test< 17 >( );
    allGood = allGood && test< 18 >( );
    allGood = allGood && test< 19 >( );
    allGood = allGood && test< 20 >( );
    allGood = allGood && test< 21 >( );
    allGood = allGood && test< 22 >( );
    allGood = allGood && test< 23 >( );
    allGood = allGood && test< 24 >( );
    allGood = allGood && test< 25 >( );
    allGood = allGood && test< 26 >( );
    allGood = allGood && test< 27 >( );
    allGood = allGood && test< 28 >( );
    allGood = allGood && test< 29 >( );
    allGood = allGood && test< 30 >( );
    allGood = allGood && test< 31 >( );
    allGood = allGood && test< 32 >( );
    allGood = allGood && test< 33 >( );
    allGood = allGood && test< 34 >( );
    allGood = allGood && test< 35 >( );
    allGood = allGood && test< 36 >( );
    allGood = allGood && test< 37 >( );
    allGood = allGood && test< 38 >( );
    allGood = allGood && test< 39 >( );
    allGood = allGood && test< 40 >( );
    allGood = allGood && test< 41 >( );
    allGood = allGood && test< 42 >( );
    allGood = allGood && test< 43 >( );
    allGood = allGood && test< 44 >( );
    allGood = allGood && test< 45 >( );
    allGood = allGood && test< 46 >( );
    allGood = allGood && test< 47 >( );
    allGood = allGood && test< 48 >( );
    allGood = allGood && test< 49 >( );
    allGood = allGood && test< 50 >( );
    allGood = allGood && test< 51 >( );
    allGood = allGood && test< 52 >( );
    allGood = allGood && test< 53 >( );
    allGood = allGood && test< 54 >( );
    allGood = allGood && test< 55 >( );
    allGood = allGood && test< 56 >( );
    allGood = allGood && test< 57 >( );
    allGood = allGood && test< 58 >( );
    allGood = allGood && test< 59 >( );
    allGood = allGood && test< 60 >( );
    allGood = allGood && test< 61 >( );
    allGood = allGood && test< 62 >( );
    allGood = allGood && test< 63 >( );
    allGood = allGood && test< 64 >( );
#endif

    std::cout << "#### Testing All Counters not sorted" << std::endl;
    MONITOR_PRINT_ALL( monitorShellLog, true )
    
    std::cout << "#### Testing All Counters not sorted" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog );

    std::cout << "#### Testing All Counters with sorting" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog, true );

    std::cout << "#### Testing All Counters not sorted" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog );

    std::cout << "#### Testing All Counters with sorting" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog, true );

    std::cout << "#### Testing Single Counter" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog, "operatorTime43" );

    std::cout << "#### Testing All Counter" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog );

    std::cout << "#### Testing Single Parameter" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog, "operatorParam8" );

    std::cout << "#### Testing All Parameters" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog );

    std::cout << "#### Testing Print All" << std::endl;
    MONITOR_PRINT_ALL( monitorShellLog )

    std::cout << "overall: " << equality_check::ok_str(allGood) << std::endl;

#ifdef MSV_USE_MONITORING
    std::cout << "Monitoring was ENABLED" << std::endl;
#else
    std::cout << "Monitoring was DISABLED" << std::endl;
#endif
    
    return !allGood;
}
