#ifndef MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H
#define MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H


//Interface Includes
//#  define EXTDIR <vector
#    include <vector/general_vector.h>
#    include <vector/primitives/calc.h>
#    include <vector/primitives/compare.h>
#    include <vector/primitives/create.h>
#    include <vector/primitives/extract.h>
#    include <vector/primitives/io.h>
#    include <vector/primitives/logic.h>
#    include <vector/primitives/manipulate.h>
//#  undef EXTDIR

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

#ifdef AVXTWO
//#  define EXTDIR <vector/simd/avx2
#    include <vector/simd/avx2/extension_avx2.h>
#    include <vector/simd/avx2/primitives/calc_avx2.h>
#    include <vector/simd/avx2/primitives/compare_avx2.h>
#    include <vector/simd/avx2/primitives/create_avx2.h>
#    include <vector/simd/avx2/primitives/extract_avx2.h>
#    include <vector/simd/avx2/primitives/io_avx2.h>
#    include <vector/simd/avx2/primitives/logic_avx2.h>
#    include <vector/simd/avx2/primitives/manipulate_avx2.h>
//#  undef EXTDIR
#endif

#ifdef SSE
//#  define EXTDIR <vector/simd/sse
#    include <vector/simd/sse/extension_sse.h>
#    include <vector/simd/sse/primitives/calc_sse.h>
#    include <vector/simd/sse/primitives/compare_sse.h>
#    include <vector/simd/sse/primitives/create_sse.h>
#    include <vector/simd/sse/primitives/extract_sse.h>
#    include <vector/simd/sse/primitives/io_sse.h>
#    include <vector/simd/sse/primitives/logic_sse.h>
#    include <vector/simd/sse/primitives/manipulate_sse.h>
//#  undef EXTDIR
#endif

//#define EXTDIR <vector/scalar
#  include <vector/scalar/extension_scalar.h>
#  include <vector/scalar/primitives/calc_scalar.h>
#  include <vector/scalar/primitives/compare_scalar.h>
#  include <vector/scalar/primitives/create_scalar.h>
#  include <vector/scalar/primitives/extract_scalar.h>
#  include <vector/scalar/primitives/io_scalar.h>
#  include <vector/scalar/primitives/logic_scalar.h>
#  include <vector/scalar/primitives/manipulate_scalar.h>
//#undef EXTDIR


#endif //MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H
