\page VectorPrimitives Vector Primitives

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
included into the code via 

<div class=userCode>
~~~{.cpp}
#include <vector/simd/sse/extension_sse.h>
~~~
</div>
The general interface is included into the operator code via

<div class=morphStoreDeveloperCode>
~~~{.cpp}
#include <vector/general_vector_extension.h>
~~~
</div>

The primitive groups can be found in 
{Morphstore/Engine/inlcude/}vector/simd/{vector extension}/primitives/{primitive group}_{vector extension}.h.

<div class=new>Instead of including all the individual headers, vector/vector_extension_structs.h contains all headers for the extensions.
Note that the corresponding flags for the usable vector extensions have to be provided when building a source using these headers. Otherwise, only the scalar versions will be available.
 <div class=userCode>
~~~{.cpp}
#include <vector/vector_extension_structs.h>
~~~
</div>
</div>

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

<div class=new>Instead of including all the individual headers, vector/vector_primitives.h contains all primitives and primitive interfaces.
Note that the corresponding flags for the usable vector extensions have to be provided when building a source using these headers.
Otherwise, only the scalar versions will be available.
<div class=morphStoreDeveloperCode>
~~~{.cpp}
#include <vector/vector_primitives.h>
~~~
</div>
</div>


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
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
namespace morphstore {

    using namespace vectorlib;
    template<class VectorExtension>
      
      void my_operator(...) {
        using namespace vectorlib;

        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        //Implement your code using vector primitives here
      }
}
~~~
</div>    
 
<div class=userCode>     
main.cpp
~~~{.cpp} 
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include "my_operator.h"

int main(){  

  using namespace vectorlib;
  using namespace morphstore;
 
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
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

namespace morphstore {

    using namespace vectorlib;
    template<class VectorExtension>
      
      int my_operator(int number) {
      
          using namespace vectorlib;
  
          IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
          
          vector_t vec = set1<VectorExtension, vector_base_type_size_bit::value>(number);
          return hadd<VectorExtension, vector_base_type_size_bit::value>::apply(vec);
          
      }
}
~~~
</div>    
 
<div class=userCode> 
main.cpp    
~~~{.cpp} 
#include <iostream>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include "my_operator.h"

int main(){  
  
  using namespace vectorlib;
  using namespace morphstore;
   
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

Of course, you can also use <i>#ifdef SSE</i>, but here we assume that any decent non-exotic system is equipped with SSE. 
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

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include "my_operator.h"

int main(){  
  
  using namespace vectorlib;
  using namespace morphstore;

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


More Information
================

For more information on the individual primitive groups, refer to one of the following pages:

[TOC]
\subpage PageSubTopicA

\subpage PageSubTopicB

\subpage PageSubTopicC

\subpage PageSubTopicD

\subpage PageSubTopicE

\subpage PageSubTopicF

\subpage PageSubTopicG

<div style="text-align:center;"> <b>prev:</b> \ref veclib   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicA </div>

\page PageSubTopicA I/O Primitives 

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
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>
      
      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void my_operator(uint64_t * dataPtr) {
      ...
        vector_t dataVector = load<VectorExtension,iov::ALIGNED, vector_size_bit::value>( dataPtr )
      ...
      }
~~~
</div>

<div style="text-align:center;"> <b>prev:</b> \ref VectorPrimitives   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicB </div>

\page PageSubTopicB Comparison Primitives 

Comparison primitives compare the content of two vector registers on a base type granularity, e.g. consider two 256-bit registers filled with 64-bit 
unsigned integers. A comparison for equality would do the following:

<pre>
vector register A:    4 | 8 | 17 | 12
vector register B:    3 | 8 |  0 | 12
-------------------------------------
A==B?                 0   1    0    1
</pre>

Thus, the result of all comparisons are a bitmask indicating whether the comparison of the corresponding elements of two vectors is true or not. A call 
of a comparison for equality looks like this:

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function() {
      ...
        vector_mask_t result = equal<VectorExtension,VectorExtension::vector_helper_t::granularity::value>::apply (
            vectorA,
            vectorB
         );
      ...
      }
~~~
</div> 

Apart from <i>equal</i>, there are more comparison primitives, namely <i>less, lessequal, greater, greaterequal, </i>and <i>count_matches</i>.

<div style="text-align:center;"> <b>prev:</b> \ref PageSubTopicA   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicC </div>


\page PageSubTopicC Calculation Primitives 

Like the comparison primitives, the binary calculation primitives take two vectors and compute the result on a base type granularity. But for calculations, 
the result is a another vector, not a mask. For instance, an addition does the following:


<pre>
vector register A:    4 | 8 | 17 | 12
vector register B:    3 | 8 |  0 | 12
-------------------------------------
A+B                   7 |16 | 17 | 24
</pre>

The according source code looks like the following:

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function() {
      ...
        vector_t result = add<VectorExtension,VectorExtension::vector_helper_t::granularity::value>::apply (
            vectorA,
            vectorB
         );
      ...
      }
~~~
</div> 

Other primitives for binary calculations are subtract <i>(sub)</i>, multiply <i>(mul)</i>, modulo <i>(mod)</i>, and divide <i>(div)</i>.
The shift primitives <i>shift_left_individual</i> and <i>shift_right_individual</i> shift the elements of the first given vector by the
corresponding amount given in the second vector.

Unary calculations take only one vector, and depending on the operation, additional parameters. The following example shows the usage 
if the <i>inv</i> primitive, which change sthe sign of the elements of a vector register:  

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function() {
      ...
        //changes the sign of all values in VectorA
        vector_t result= inv<VectorExtension>::apply(VectorA);
      ...
      }
~~~
</div>

Another calculation primitive, which takes only one vector as an argument is horizontal add <i>(hadd)</i>.
There are also shift primitives. <i>Shift_left</i> and <i>shift_right</i> take an integer indicating how far the elements in the vector are shifted.

<div style="text-align:center;"> <b>prev:</b> \ref PageSubTopicB   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicD </div>

 
\page PageSubTopicD Create Primitives 

There are 3 create primitives:

- set1: Sets all elements of a vector register to a given value
- set_sequence: Fills a vector register with a sequence of integer values, useful for creating and incrementing index lists
- set: Sets the elements of a vector register to the given values. The __use of this primitive is highly discouraged__ because the number of given values 
depends on the vector size and the size of the base type, which breaks with the concept of universally applicable primitives.

The <i>set1</i> and <i>set_sequence</i> primitives can be used as shown in the following code snippet:

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function() {
      ...
      //A vector filled with 5, 5, 5, 5,...
      vector_t vec_five = vectorlib::set1<VectorExtension, vector_base_t_granularity::value>(5);
      //A vector filled with 0, 2, 4, 6, ...
      vector_t vec_sequence = vectorlib::set_sequence<VectorExtension, vector_base_t_granularity::value>(0,2);
      ...
      }
~~~
</div>
      
<div style="text-align:center;"> <b>prev:</b> \ref PageSubTopicC   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicE </div>


\page PageSubTopicE Extract Primitives 

There is only one extract primitive: <i>extract_value</i>, which extracts a single value from a register. The type of the extracted value is <i>base_t</i>,
which internally casts to the chosen base type of the processing style.

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function(vector_t dataVector) {
      ...
      //extracts the first value (value with index 0) from a register
      base_t value0 = extract_value<VectorExtension,vector_base_t_granularity::value>(dataVector, 0);
      ...
      }
~~~
</div>     

<div style="text-align:center;"> <b>prev:</b> \ref PageSubTopicD   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicF </div>

\page PageSubTopicF Logic Primitives 

Logic primitives perform bitwise comparisons between two vectors and return another vector with the result of the comparison. 
Currently, there are primitives for bitwise AND and bitwise OR. They can be used like in the following example:

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function(vector_t dataVector) {
      ...
      vector_t result_and = bitwise_and<VectorExtension, vector_size_bit::value>(vectorA, vectorB);
      vector_t result_or  = bitwise_or <VectorExtension, vector_size_bit::value>(vectorA, vectorB);
      ...
      }
~~~
</div>     
               

<div style="text-align:center;"> <b>prev:</b> \ref PageSubTopicE   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref PageSubTopicG </div>

\page PageSubTopicG Manipulate Primitives 

Manipulate primitives manipulate the content of a vector register on a vector size level.
Currently, there is only one primitive in this group: <i>rotate</i>, which is used to rotate the elements in a vector by one element in each call.
For instance, assume there is a register filled with the values { 3, 2, 1, 0}. After calling rotate, the content of the register is { 2, 1, 0, 3}.

<div class=morphStoreDeveloperCode>
my_operator.h
~~~{.cpp}
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

template<class VectorExtension>

      IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
      void some_function() {
      ...
      vector_t result = rotate<VectorExtension, vector_size_bit::value>(vectorA);
      ...
      }
~~~
</div>     


<div style="text-align:center;"> <b>prev:</b> \ref PageSubTopicF   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>Next: </b>\ref primitiveTable </div>