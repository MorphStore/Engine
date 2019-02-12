/**
 * @file storage_test.cpp
 * @brief Brief description
 * @author Patrick Damme
 * @author Johannes Pietrzyk
 * @todo TODOS?
 */

#include "../../../include/core/memory/mm_glob.h"
#include "../../../include/core/morphing/format.h"
#include "../../../include/core/storage/column.h"

#include <cstddef>
#include <cstdio>
#include <vector>
#include <iostream>

using namespace morphstore;

void fillColumn( column< uncompr_f > * p_Col, size_t p_CountValues ) {
    uint64_t * const data = p_Col->get_data( );
    for( unsigned i = 0; i < p_CountValues; i++ )
        data[ i ] = i;
    p_Col->set_count_values( p_CountValues );
    p_Col->set_size_used_byte( p_CountValues * sizeof( uint64_t ) );
}

void printColumn( const column< uncompr_f > * p_Col ) {
    const size_t countValues = p_Col->get_count_values( );
    const size_t countValuesPrint = std::min(
        static_cast< size_t >( 10 ),
        countValues / 2
    );
    const uint64_t * const data = p_Col->get_data( );
    for( unsigned i = 0; i < countValuesPrint; i++ )
        std::cout << data[ i ] << ',';
    std::cout << " ... ";
    for( unsigned i = countValues - countValuesPrint; i < countValues; i++ )
        std::cout << data[ i ] << ',';
    std::cout << "done." << std::endl;
}

int main( void ) {
    const size_t countValues = 100 * 1000 * 1000;
    const size_t sizeAllocateByte = countValues * sizeof( uint64_t );
    
    std::cout << "Testing an ephemeral column:" << std::endl;
    auto colEphi = new column< uncompr_f >( sizeAllocateByte );
    fillColumn( colEphi, countValues );
    printColumn( colEphi );
    
    std::cout << "Testing a perpetual column:" << std::endl;
    auto colPerp = column< uncompr_f >::create_perpetual_column( sizeAllocateByte );
    fillColumn( colPerp, countValues );
    printColumn( colPerp );
    
    return 0;
}