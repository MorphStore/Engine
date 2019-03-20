/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>
//#include <core/operators/aggregation/summation.h>
#include <core/operators/scalar/agg_sum_uncompr.h>
#include <core/operators/vectorized/agg_sum_uncompr.h>

#include <iostream>


#define TEST_DATA_COUNT 100000000

using namespace morphstore;

void init_data( column< uncompr_f > * const dataColumn ) {
   uint64_t * data = dataColumn->get_data( );
   size_t const count = TEST_DATA_COUNT / sizeof( uint64_t );
   for( size_t i = 0; i < count; ++i ) {
      data[ i ] = static_cast< uint64_t >( 1 );
   }
   dataColumn->set_meta_data( count, TEST_DATA_COUNT );
}

int main( void ) {
    column< uncompr_f > * testDataColumn = column<uncompr_f>::create_global_column(TEST_DATA_COUNT);
    init_data(testDataColumn);

    std::cout << "Start scalar aggregation...\n";
    auto sum_aggscalar_result=agg_sum<processing_style_t::scalar>( testDataColumn );//Do aggregation
    std::cout << "Done!\n";
    std::cout << "Should be "<< TEST_DATA_COUNT / sizeof( uint64_t ) << ". is: " << *((uint64_t*)(sum_aggscalar_result->get_data())) << "\n";
    
    std::cout << "Start aggregation with 128 bit registers...\n";
    auto sum_agg128_result=agg_sum_128<processing_style_t::scalar>( testDataColumn );//Do aggregation
    std::cout << "Done!\n";
    std::cout << "Should be "<< TEST_DATA_COUNT / sizeof( uint64_t ) << ". is: " << *((uint64_t*)(sum_agg128_result->get_data())) << "\n";
    
    std::cout << "Start aggregation with 256 bit registers...\n";
    auto sum_agg256_result=agg_sum_256<processing_style_t::scalar>( testDataColumn );//Do aggregation
    std::cout << "Done!\n";
    std::cout << "Should be "<< TEST_DATA_COUNT / sizeof( uint64_t ) << ". is: " << *((uint64_t*)(sum_agg256_result->get_data())) << "\n";
    
    return 0;
}