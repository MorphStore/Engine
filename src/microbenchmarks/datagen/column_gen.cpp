#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/persistence/binary_io.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/helper.h>
#include <string>
#include <iostream>
#include <filesystem>

constexpr size_t count_values( size_t p_Size ) {
   return p_Size / sizeof( uint64_t );
}


int main( void ) {

   using namespace morphstore;

   string path = std::filesystem::current_path();
   uint64_t lower, upper;
   std::cout << "Lower: " << std::flush;
   std::cin >> lower;
   std::cout << "Upper: " << std::flush;
   std::cin >> upper;

   std::vector< size_t > selectivities = { 25, 50, 75 };
   std::vector< size_t > countValues = {
      count_values( 8_MB ),
      count_values( 16_MB ),
      count_values( 64_MB ),
      count_values( 128_MB ),
      count_values( 1_GB ),
      count_values( 4_GB )
   };

   for( auto sel : selectivities ) {
      for( auto count : countValues ) {
         const column< uncompr_f > * dataCol =
            generate_specific_selective (
               count,
               lower,
               upper,
               sel
         );
         binary_io< uncompr_f >::store(
            dataCol,
            dataPath + "/data_sel_" + sel + "_count_" + count + "_lower_" + lower + "_upper_" + upper + ".bin"
         );
      }
   }
   return 0;

}
