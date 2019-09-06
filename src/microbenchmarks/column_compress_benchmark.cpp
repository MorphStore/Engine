#include <core/memory/mm_glob.h>
#include <core/memory/noselfmanaging_helper.h>
#include <core/morphing/format.h>
#include <core/morphing/dynamic_vbp.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/operators/scalar/select_uncompr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/variant_executor.h>

#include <vector/primitives/compare.h>
#include <vector/scalar/extension_scalar.h>
#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>

#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/logic_sse.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using namespace morphstore;
using namespace vector;

int main() {
    const unsigned t_bw = 1;
    unsigned countValues = 
                64 << 20;


    //select_handwritten_buffer_compr_out<8>(origCol, val)
    using in_f = uncompr_f;
    using out_f = static_vbp_f<t_bw, 1>;

    //const size_t inDataCount = origCol->get_count_values();
    //const uint64_t * const inData = origCol->get_data();

    // If no estimate is provided: Pessimistic allocation size (for
    // uncompressed data), reached only if all input data elements pass the
    // selection.

    const unsigned ARRAY_SIZE = 5;
    column<in_f>* cols[ARRAY_SIZE];

    for (unsigned i = 0; i < ARRAY_SIZE; i++)
    {
        cols[i] = const_cast<column<in_f>*>(
            generate_exact_number(
                countValues,
                10 << 10,
                123,
                12345) );
    }
    const column<in_f>** actualCols = const_cast<const column<in_f>**>(cols);
    std::cout << "Generated numbers" << std::endl;

    column<out_f>* outcols[ARRAY_SIZE];
    for (unsigned i = 0; i < ARRAY_SIZE; i++)
    {
        outcols[i] = const_cast<column<out_f>*>(
            morph<
                scalar<v64 <uint64_t>>,
                out_f,
                in_f
                    >( actualCols[i] ) );
    }
    const column<out_f>** outPosCols = const_cast<const column<out_f>**>(outcols);

    auto originalSize = outPosCols[0]->get_size_used_byte();
    std::cout << "Column compressed bytes for " << countValues << " values: " << outPosCols[0]->get_size_compr_byte();
    std::cout << ", bytes used: " << outPosCols[0]->get_size_used_byte() << std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    for (unsigned i = 0; i < ARRAY_SIZE; i++) {
        const_cast<column<out_f>*>(outPosCols[i])->reallocate();
    }
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Needed " << elapsed_seconds.count() << " for reallocing " << originalSize << " bytes" << std::endl;

    return 0;
}
