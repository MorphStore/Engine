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

void fillColumn( morphstore::storage::column< morphstore::morphing::format::UNCOMPR > * p_Col, size_t p_CountValues ) {
    uint64_t * const data = reinterpret_cast< uint64_t * >( p_Col->data( ) );
    for( unsigned i = 0; i < p_CountValues; i++ )
        data[ i ] = i;
    p_Col->count_values( p_CountValues );
    p_Col->size_used_byte( p_CountValues * sizeof( uint64_t ) );
}

void printColumn( const morphstore::storage::column< morphstore::morphing::format::UNCOMPR > * p_Col ) {
    using namespace std;
    
    const size_t countValues = p_Col->count_values( );
    const size_t countValuesPrint = min(
        static_cast< size_t >( 10 ),
        countValues / 2
    );
    const uint64_t * const data = reinterpret_cast< const uint64_t * >( p_Col->data( ) );
    for( unsigned i = 0; i < countValuesPrint; i++ )
        cout << data[ i ] << ',';
    cout << " ... ";
    for( unsigned i = countValues - countValuesPrint; i < countValues; i++ )
        cout << data[ i ] << ',';
    cout << "done." << endl;
}

int main( void ) {
    using namespace morphstore::morphing;
    using namespace morphstore::storage;
    using namespace std;
    
    const size_t countValues = 100 * 1000 * 1000;
    const size_t sizeAllocateByte = countValues * sizeof( uint64_t );
    
    cout << "Testing an ephemeral column:" << endl;
    column< format::UNCOMPR > * colEphi = new column< format::UNCOMPR >(sizeAllocateByte);
    fillColumn( colEphi, countValues );
    printColumn( colEphi );
    
    cout << "Testing a perpetual column:" << endl;
    column< format::UNCOMPR > * colPerp = column< format::UNCOMPR >::createPerpetualColumn(sizeAllocateByte);
    fillColumn( colPerp, countValues );
    printColumn( colPerp );
    
    return 0;
}