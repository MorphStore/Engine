VectorLib
=========
All files associated with vectorization should be located in ${PROJECT_HOME}/include/vector/ and reside in `namespace vector`.

General remarks
---------------
Through the variety of existing minor versions of vector extensions provided by Intel (SSE2/3/4.1/4.2,...) only the major extensions (in terms of applicability) is provided (and subsumes the related versions).

E.g.: 
- SSE = { SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2 }
- AVX2 = { AVX, AVX2 }

####Exception
Thus AVX-512 has special subsets which may or may not appear together onto a processor, a distinction where be made:

- AVX512 = { AVX-512, AVX-512F }
- AVX512BW = AVX-512BW
- AVX512DQ = AVX-512DQ 
- AVX512CD = AVX-512CD

Conceptual remarks
------------------
We try to abstract vectorized computation with its low-level intrinsics api and execution models from the general C++-code.

####Vector Registers
Thus there is an abstract vector-register struct (vector/general_vector.h):
```
1    template<uint16_t BitWidth, typename T>
2    struct vector_reg {
3      static constexpr uint16_t size_b = BitWidth / 8;
4      static constexpr uint16_t element_count = size_b / sizeof(T);
5   };
```
This helper struct can be used for simple arithmetic tasks and facilitates templated access.
A Vector register holding up to 128-bit data is a specialization of `vector_reg`:
```
1    template<typename T>
2    using v128 = vector_reg<128, T>;
```

####Vector Extensions
To realize template function wrappers for low-level intrinsics we use another helper struct per available vector extension (vector/simd/extension.h):
```
 1   template<typename T>
 2   struct sse< v128< T > > {
 3      static_assert(std::is_arithmetic<T>::value, "Base type of vector register has to be arithmetic.");
 4      using vector_helper_t = v128<T>;
 5
 6      using vector_t =
 7      typename std::conditional<
 8         std::is_integral<T>::value,    // if T is integer
 9         __m128i,                       //    vector register = __m128i
10         typename std::conditional<
11            std::is_same<float, T>::value, // else if T is float
12            __m128,                       //    vector register = __m128
13            __m128d                       // else [T == double]: vector register = __m128d
14         >::type
15      >::type;
16
17      using size = std::integral_constant<size_t, sizeof(vector_t)>;
18      using mask_t = uint16_t;
19   };
```
First of all we are focusing on arithmetic types (see line 3 ). To query the number of elements or the size of a vector register the helper struct `vector_helper_t` can be used (see line 4).
In addition the depicted sse-struct provides a vector type `vector_t`. The underlying type of `vector_t` depends on the specified basetype T. For integral types (like `uint8_t, uint32_t, uint64_t,...`), `vetor_t` is `__m128i` (see line 9). If the basetype is float then `vector_t` will be `__m128`, `__m128d` for double respectively.

Implementing
------------------

####Primitives
In general a primitives looks like the following:
```
    template<class VectorExtension>
    struct primitive;
```
`VectorExtension` has to be a template struct like sse. This general template should enable a plethora of possible specialization.

If partial specialization regarding the underlying type, the size of the vector register or the used vector extension should be realized,
a partial specialization can be used. **Watch out:** partial specialization is **not** possible when using a global function. Use a struct instead (see below). 
One of the main drawbacks with structs is that c++-standard prohibits static functors ( `static operator()` ). To avoid the need of instantiating the particular
struct aside passing a `this` pointer and so on an additional member method has to be used. Maybe this can be shortened for ease of use.

If a primitive shall be fully specialized, an ordinary global function can be used.

```
 1    template<typename T, int IOGranularity>
 2    struct load<sse<v128<T>>,iov::ALIGNED, IOGranularity> {
 3        template< typename U = T, typename std::enable_if< std::is_integral< U >::value, int >::type = 0 >
 4        MSV_CXX_ATTRIBUTE_INLINE
 5        static typename sse< v128< U > >::vector_t
 6        apply( U const * const p_DataPtr ) {
 7            trace( "[VECTOR] - Loading aligned integer values into 128 Bit vector register." );
 8            return _mm_load_si128(reinterpret_cast<typename sse< v128< U > >::vector_t const *>(p_DataPtr));
 9        }
10
11        template< typename U = T, typename std::enable_if< std::is_same< float, U >::value, int >::type = 0 >
12        MSV_CXX_ATTRIBUTE_INLINE
13        static typename sse< v128< U > >::vector_t
14        apply( U const * const p_DataPtr ) {
15            trace( "[VECTOR] - Loading aligned float values into 128 Bit vector register." );
16            return _mm_load_ps(reinterpret_cast< U const * >(p_DataPtr));
17        }
18
19        template< typename U = T, typename std::enable_if< std::is_same< double, U >::value, int >::type = 0 >
20        MSV_CXX_ATTRIBUTE_INLINE
21        static typename sse< v128< U > >::vector_t
22        apply( U const * const p_DataPtr ) {
23            trace( "[VECTOR] - Loading aligned double values into 128 Bit vector register." );
24            return _mm_load_pd(reinterpret_cast< U const * >(p_DataPtr));
25        }
26    };
```

This is an example for implementing a partial specialized load primitive. Line 3-9 cover all integral basetypes, Line 11-25 all floating point basetypes.

####Usage
```
1    using namespace vector;
2    uint32_t * const data = (uint32_t*)_mm_malloc( 128, 16 );
3    typename sse< v128< uint32_t > >::vector_t a =
4        load< sse< v128< uint32_t > >, iov::ALIGNED, 128 >::apply( data );
5    typename sse< v128< double > >::vector_t b =
6        load< sse< v128< double > >, iov::UNALIGNED, 128 >::apply( reinterpret_cast< double * >( data ) );
7    _mm_free( data );
```

Important Files
---------------
- vector/
  - general_vector.h --> Definitions for vector registers.
  - primitives/
    - io.h --> Abstract definitions for I/O related functions.
    - ...
  - simd/
    - sse/
      - extension_sse.h --> Definition for vector extension struct SSE.
      - primitives/
        - io_sse.h --> Implementations of I/O related functions for SSE
    - avx2/
      - ...
    - avx512/
      - ...
  



TODO
----
1. Realize detection of available hardware vector extensions and provide proper preprocessor variables like `MSV_VECTORLIB_SSE` if SSE is enable e.g.
2. Discuss possibilities to shorten static member function of partial specialized structs (aliasing?).
3. Realize a dependent include (with respect to existing hardware extensions)
4. Implement:
    - I/O
        - Store
        - Gather
        - Scatter
    - ...