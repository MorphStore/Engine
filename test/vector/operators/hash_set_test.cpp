//
// Created by jpietrzyk on 20.05.19.
//

#include <vector/primitives/logic.h>
#include <vector/primitives/io.h>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/logic_avx2.h>
#include <vector/simd/avx2/primitives/io_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/compare_avx2.h>
#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>

#include <vector/datastructures/set_utils.h>
#include <vector/datastructures/hash_set.h>
#include <vector/datastructures/set_strategies.h>
#include <vector/complex/hash.h>


#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>


#include <vector>
#include <algorithm>


int main( void ) {

   using namespace vector;
   using namespace morphstore;

   size_t const dataCount = 128;
   size_t const loadfactor = 60;
//   size_t const mapCount = (dataCount * 100 / loadfactor);

   auto col1 = generate_sorted_unique(dataCount);
   /*auto col2 = new column<uncompr_f>( mapCount* sizeof( uint64_t));
   col2->set_meta_data(mapCount, mapCount*sizeof(uint64_t));
   uint64_t * d = col1->get_data();

//   hash_set_lpcs< avx2<v256<uint64_t>>, multiply_mod_hash, size_policy_set::ARBITRARY, 60> hs( col1 );
   scalar_key_vectorized_linear_search<
      sse<v128<uint64_t>>,
      avx2<v256<uint64_t>>,
      multiply_mod_hash,
      size_policy_set::ARBITRARY
   >::build_batch( d, col2->get_data(), dataCount, mapCount);
*/

   hash_set<
      avx2<v256<uint64_t>>,
      multiply_mod_hash,
      size_policy_set::ARBITRARY,
      scalar_key_vectorized_linear_search,
      60 >
   hs( col1->get_count_values() );
   hs.build< uncompr_f, sse<v128<uint64_t>> >( col1 );

   auto col2 = hs.get_data();

   std::vector<uint64_t> myvector( col2, col2 + hs.get_bucket_count());
   std::sort (myvector.begin(), myvector.end());
   auto col3 = make_column( myvector.data(),  hs.get_bucket_count() );

   print_columns(
      print_buffer_base::decimal,
      col1,
      col3,
      "data",
      "hashset"
   );

   return 0;
}