#ifndef MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H
#define MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H


//Interface Includes
#  define EXTDIR <vector
#    include EXTDIR/general_vector.h>
#    include EXTDIR/primitives/calc.h>
#    include EXTDIR/primitives/compare.h>
#    include EXTDIR/primitives/create.h>
#    include EXTDIR/primitives/extract.h>
#    include EXTDIR/primitives/io.h>
#    include EXTDIR/primitives/logic.h>
#    include EXTDIR/primitives/manipulate.h>
#  undef EXTDIR

#ifdef AVX512
#  define EXTDIR <vector/simd/avx512
#    include EXTDIR/extension_avx512.h>
#    include EXTDIR/primitives/calc_avx512.h>
#    include EXTDIR/primitives/compare_avx512.h>
#    include EXTDIR/primitives/create_avx512.h>
#    include EXTDIR/primitives/extract_avx512.h>
#    include EXTDIR/primitives/io_avx512.h>
#    include EXTDIR/primitives/logic_avx512.h>
#    include EXTDIR/primitives/manipulate_avx512.h>
#  undef EXTDIR
#endif

#ifdef AVX2
#  define EXTDIR <vector/simd/avx2
#    include EXTDIR/extension_avx2.h>
#    include EXTDIR/primitives/calc_avx2.h>
#    include EXTDIR/primitives/compare_avx2.h>
#    include EXTDIR/primitives/create_avx2.h>
#    include EXTDIR/primitives/extract_avx2.h>
#    include EXTDIR/primitives/io_avx2.h>
#    include EXTDIR/primitives/logic_avx2.h>
#    include EXTDIR/primitives/manipulate_avx2.h>
#  undef EXTDIR
#endif

#ifdef SSE
#  define EXTDIR <vector/simd/sse
#    include EXTDIR/extension_sse.h>
#    include EXTDIR/primitives/calc_sse.h>
#    include EXTDIR/primitives/compare_sse.h>
#    include EXTDIR/primitives/create_sse.h>
#    include EXTDIR/primitives/extract_sse.h>
#    include EXTDIR/primitives/io_sse.h>
#    include EXTDIR/primitives/logic_sse.h>
#    include EXTDIR/primitives/manipulate_sse.h>
#  undef EXTDIR
#endif

#define EXTDIR <vector/scalar
#  include EXTDIR/extension_scalar.h>
#  include EXTDIR/primitives/calc_scalar.h>
#  include EXTDIR/primitives/compare_scalar.h>
#  include EXTDIR/primitives/create_scalar.h>
#  include EXTDIR/primitives/extract_scalar.h>
#  include EXTDIR/primitives/io_scalar.h>
#  include EXTDIR/primitives/logic_scalar.h>
#  include EXTDIR/primitives/manipulate_scalar.h>
#undef EXTDIR


#endif //MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H
