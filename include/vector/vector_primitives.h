#ifndef MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H
#define MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H


//Interface Includes
#include <vector/vector_extension_structs.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/create.h>
#include <vector/primitives/extract.h>
#include <vector/primitives/io.h>
#include <vector/primitives/logic.h>
#include <vector/primitives/manipulate.h>
//#  undef EXTDIR

#ifdef AVX512
#  include <vector/simd/avx512/primitives/calc_avx512.h>
#  include <vector/simd/avx512/primitives/compare_avx512.h>
#  include <vector/simd/avx512/primitives/create_avx512.h>
#  include <vector/simd/avx512/primitives/extract_avx512.h>
#  include <vector/simd/avx512/primitives/io_avx512.h>
#  include <vector/simd/avx512/primitives/logic_avx512.h>
#  include <vector/simd/avx512/primitives/manipulate_avx512.h>
#endif

#ifdef AVXTWO
#  include <vector/simd/avx2/primitives/calc_avx2.h>
#  include <vector/simd/avx2/primitives/compare_avx2.h>
#  include <vector/simd/avx2/primitives/create_avx2.h>
#  include <vector/simd/avx2/primitives/extract_avx2.h>
#  include <vector/simd/avx2/primitives/io_avx2.h>
#  include <vector/simd/avx2/primitives/logic_avx2.h>
#  include <vector/simd/avx2/primitives/manipulate_avx2.h>
#endif

#ifdef SSE
#  include <vector/simd/sse/primitives/calc_sse.h>
#  include <vector/simd/sse/primitives/compare_sse.h>
#  include <vector/simd/sse/primitives/create_sse.h>
#  include <vector/simd/sse/primitives/extract_sse.h>
#  include <vector/simd/sse/primitives/io_sse.h>
#  include <vector/simd/sse/primitives/logic_sse.h>
#  include <vector/simd/sse/primitives/manipulate_sse.h>
#endif

#ifdef NEON
#  include <vector/simd/neon/primitives/calc_neon.h>
#  include <vector/simd/neon/primitives/compare_neon.h>
#  include <vector/simd/neon/primitives/create_neon.h>
#  include <vector/simd/neon/primitives/extract_neon.h>
#  include <vector/simd/neon/primitives/io_neon.h>
#  include <vector/simd/neon/primitives/logic_neon.h>
#  include <vector/simd/neon/primitives/manipulate_neon.h>
#endif

#include <vector/scalar/primitives/calc_scalar.h>
#include <vector/scalar/primitives/compare_scalar.h>
#include <vector/scalar/primitives/create_scalar.h>
#include <vector/scalar/primitives/extract_scalar.h>
#include <vector/scalar/primitives/io_scalar.h>
#include <vector/scalar/primitives/logic_scalar.h>
#include <vector/scalar/primitives/manipulate_scalar.h>
//#undef EXTDIR


#endif //MORPHSTORE_VECTOR_PRIMITIVES_PRIMITIVES_H
