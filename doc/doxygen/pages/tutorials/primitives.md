\page PageMainTopic Vector Primitives
[TOC]
\subpage PageSubTopicA

\subpage PageSubTopicB

Vector Primitives
=================
Vector primtives are functionalities, which are frequently used in most operators, e.g. load and store operations. 
However, different vector extensions support these operations either in a different way or not at all, which would 
require a different operator implementation for every extension. For this reason, vector primitives are introduced. They 
provide a common interface for all (implemented) vector extensions, such that an operator can be implemented only once 
and then be instantiated with different vector extensions and vector sizes. There are different primitive groups 
providing a variety of functions, e.g. I/O operations and arithmetic.

Includes
--------
For using the primitives, the according header for each extension and for each primitive group has to be included. The 
extension header can be found in 
{Morphstore/Engine/inlcude/}vector/simd/{vector extension}/extension_{vector extension}.h, e.g. the SSE extension is 
included into the user code via 

<div class=userCode>
~~~{.cpp}
#include <vector/simd/sse/extension_sse.h>
~~~
</div>
The general interface is included into the operator code via

<div class=morphStoreDeveloperCode>
~~~{.cpp}
#include <vector/general_vector.h>
~~~
</div>

The primitive groups can be found in 
{Morphstore/Engine/inlcude/}vector/simd/{vector extension}/primitives/{primitive group}_{vector extension}.h.

###Primitive groups
There are 6 primitive groups:

- Calc: Includes arithmetic calculations, e.g. addition, subtraction, multiplication,...
- Compare: Implements element-wise comparisons between the two operators, e.g. equality, smaller than, greater than
- Create: Includes functions to create new vectors, e.g. filled with a constant or a number sequence
- Extract: Can be used to extract single values from a register
- I/O: Provides different load and store functions
- Manipulate: Manipulate the content of one vector, currently this only implements a rotate function

To include all primitve groups for SSE into user code, the include directives are the following  

<div class=userCode>
~~~{.cpp}
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/compare_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include <vector/simd/sse/primitives/extract_sse.h>
#include <vector/simd/sse/primitives/io_sse.h>
#include <vector/simd/sse/primitives/manipulate_sse.h>
~~~
</div>

The corresponding interfaces can be included into the operator code like the following:

<div class=morphStoreDeveloperCode>
~~~{.cpp}
#include <vector/primitives/calc.h>
#include <vector/primitives/compare.h>
#include <vector/primitives/create.h>
#include <vector/primitives/extract.h>
#include <vector/primitives/io.h>
#include <vector/primitives/manipulate.h>
~~~
</div>

<b>When implementing a new operator, the operator includes the generic interfaces. The concrete extension headers only have 
to be included when calling this new operator. The remainder of this tutorial shows how to implement such an operator and 
how to call it. </b>

Vector Extensions and Sizes
---------------------------
In order to use a primitive, a vector extension, a vector size, and a granularity has to be 
chosen. Currently the Intel extensions SSE, AVX2, and AVX-512 are supported. The corresponding vector sizes are v128, 
v256, and v512 (128, 256, and 512 bit). The granlarity refers to the element size in bit, which follows from the type of
the vector elements. To simplify all primitive calls, the wrapper class <i>VectorExtension</i> encapsulates an 
extension, a vector size, and a base type. Importing a boilerplate adds some useful parameters derieved from the vector
extension, e.g. <i>vector_element_count</i> which indicates the number of elements in a vector.

The following code snippets show a template for an operator implementation and how to call this operator for a specific 
extension, here for AVX2 using 256-bit registers and an integral base type. Note that all vector extension related 
functionality is in the namespace <i>vector</i>.

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/general_vector.h>

template<class VectorExtension>
      
      void my_operator(...) {
        using namespace vector;

        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        //Implement your code using vector primitives here
      }
~~~
</div>    
 
<div class=userCode>     
main.cpp
~~~{.cpp} 
#include <vector/simd/avx2/extension_avx2.h>
#include "my_operator.h"

int main(){  

  using namespace vector;
   
  my_operator<avx2<v256<uint64_t>>>( ... );
  return 0;

}
~~~
</div>

Using Primitives
----------------
The following example extends the operator template from above. It now creates a vector, broadcasts a number to all elements of this vector, 
and finally returns and displays the sum of all elements of the created vector.

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/general_vector.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>


template<class VectorExtension>
      
      int my_operator(int number) {
      
        using namespace vector;

        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        
        vector_t vec = set1<VectorExtension, vector_base_type_size_bit::value>(number);
        return hadd<VectorExtension, vector_base_type_size_bit::value>(vec);
        
      }
~~~
</div>    
 
<div class=userCode> 
main.cpp    
~~~{.cpp} 
#include <iostream>
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#include "my_operator.h"

int main(){  
  
  using namespace vector;
   
  int result = my_operator<avx2<v256<uint64_t>>>( 1 );
  
  std::cout << "Result sum: " << result << std::endl;
  
  return 0;

}
~~~
</div>

In this example, two primitives are called: set1 and hadd. 
Set1 is part of the <i>create</i> primitive group, while hadd belongs into the <i>calc</i> primitive group. Note that the 
according headers are included.

Set1 sets all elements of a new vector to a number, in this case this number is 1. We defined a 256-bit vector with 
uint64_t elements. This translates to a vector with 4 integral elements. Hence, set1 creates a vector, which is filled 
with 4 64 bit elements, each encoding the integral number 1. 

Hadd takes the result of set1 as input and computes the sum of all elements in this input vector. In this example, the 
result is 4.

This example produces the following output: 

<pre>Result sum: 4</pre>

Optional Support
----------------
When calling an operator or a primitive with a concrete vector extension, which requires anything newer than SSE, it is 
highly recommended to put this code in an <i>#ifdef</i>-block. This can be useful if several alternatives are 
possible but not all of them are available on every system. The following code shows how such blocks look like. 

<div class=userCode>     
~~~{.cpp} 
#ifdef AVXTWO
//everything that needs at least AVX 2
#endif

#ifdef AVX512
//everything that needs AVX-512
#endif
~~~
</div>

In our running example, we can create an SSE and an AVX2 variant and put the AVX2 variant into a 
<i>#ifdef</i>-block. This creates the output

<pre>
Result sum (avx2): 4
Result sum (sse): 2
</pre>

<div class=userCode>  
main.cpp   
~~~{.cpp} 
#include <iostream>

#ifdef AVXTWO
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#endif

#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include "my_operator.h"

int main(){  
  
  using namespace vector;
  int result = 0;
  
  #ifdef AVXTWO 
  result = my_operator<avx2<v256<uint64_t>>>( 1 );
  std::cout << "Result sum (avx2): " << result << std::endl;
  #endif
  
   
  result = my_operator<sse<v128<uint64_t>>>( 1 );
  std::cout << "Result sum (sse): " << result << std::endl;
  
  
  return 0;

}
~~~
</div>

Build
-----
Using the <i>build.sh</i> script enables sse support by default. If AVX2 or AVX512 are used, the script must be 
called with the flags -avxtwo or -avx512 respectively, where -avx512 enables AVX512 and AVX2 support. 

If our running example is built without any of these flags, it produces the output

<pre>
Result sum (sse): 2
</pre>  

\page PageSubTopicA I/O Primitives 

I/O Primitives
==============

The load and store primitives implement 4 functions:

- load: takes a pointer to a memory address and returns a vector
- store: takes a pointer to a memory address and a vector
- compressstore: like store but additonally takes a bit mask
- gather: takes a pointer to a memory address and a vector

Each of these functions takes 3 template arguments: vector extension, IO variant, and granularity.
While the vector extension was already used in the running example, e.g. avx2<v256<uint64_t>>, and granularity is simply 
the size of the loaded or stored elements in bit, IO variant is not treated, yet. The IO variant defines if data access 
is aligned, unaligned, or streamed. Additonally, there is an option called unalignedx, which calls a specialized load, 
which is different from the usual unaligned load in some vector extensions. All options are shown in the following code 
snippet  

<div class=morphStoreBaseCode>
~~~{.cpp}
   enum class iov {
      ALIGNED,
      UNALIGNED,
      UNALIGNEDX,
      STREAM
   };
~~~
</div>

The <b>load</b> function loads the data at a given memory address into a vector register.

The <b>store</b> function stores the content of a given vector register to a given memory address.

<b>Compress-store</b> Only stores the elements for which a bit in a bitmask is set.

<b>Gather</b> loads data from a given memory address, where the offset for each element is given in a vector register.  

The following example shows an aligned load:

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/general_vector.h>
#include <vector/primitives/io.h>

template<class VectorExtension>
      
      void my_operator(uint64_t * dataPtr) {
      ...
        load<VectorExtension,iov::ALIGNED, vector_size_bit::value>( dataPtr )
      ...
      }
~~~
</div>

<div class="ToDo">Write a little more</div>
\page PageSubTopicB Compare Primitives 

Compare Primitives
==============
<div class="ToDo">Write this</div>