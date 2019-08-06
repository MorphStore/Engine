\page helloWorld Hello World!

<a href="#setup">Create a c++ file</a><br />
<a href="#query">The Example Query</a><br />
<a href="#data">Create Test Data</a><br />
<a href="#operators">Operators</a><br />
<a href="#compile">Compile and Run</a>

<h2 id=setup></h2>
Create a c++ file
============
This page explains how to write a little test query. The full code can be found in src/examples/select_sum_query.cpp.

First, create a new *.cpp file, preferably in the folder src/examples. Include the header of the global memory manager. 
Make sure to always include it at the very beginning before any other includes. Then, include the headers <i>functional</i>,
<i>iostream</i>, and <i>random</i>. Additonally, we will be using the namespaces <i>morphstore</i> and <i>vectorlib</i>. 
Finally, create an empty main function.

<div class=userCode>
~~~{.cpp}
#include <core/memory/mm_glob.h>
#include <functional>
#include <iostream>
#include <random>

using namespace morphstore;
using namespace vectorlib;

int main( void ) {

  return 0;
}
~~~
</div>

<h2 id=query></h2>
The Example Query
=================

The SQL equivalent of the example query is the following:

~~~{sql}
SELECT SUM(baseCol1) 
FROM someUnnamedTable 
WHERE baseCol2 = 150
~~~

Obviously, we need to create two columns for this example with a numerical type: baseCol1 and baseCol2. 
The necessary operators are a select on baseCol2, a projection to find the according values in baseCol1, 
and an aggregation to sum up the resulting values.

<h2 id=data></h2>
Create Test Data
================

There is a column generator to create numerical test data with a given distribution. 
To use the data generator, we include four additional headers. Remember to put these includes _after_ 
the include of mm_glob.h.  

<div class=userCode>
~~~{.cpp}
...
#include <core/morphing/format.h>     //definition of different (compression) formats including uncompressed
#include <core/storage/column.h>      //definition of columns
#include <core/storage/column_gen.h>  //data generator
#include <core/utils/basic_types.h>   //some naming conversions
...
~~~
</div>

We will generate two columns, each having 100M unsorted, uniformly distributed values ranging from 100 to 199, and 
from 0 to 10 respectively: 

<div class=userCode>
~~~{.cpp}
...
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    const size_t countValues = 100 * 1000 * 1000;
    const column<uncompr_f> * const baseCol1 = generate_with_distr(   //create a column with uncompressed values
            countValues,                                              //100M values
            std::uniform_int_distribution<uint64_t>(100, 199),        //uniformly distibuted from 100 to 199
            false                                                     //not sorted
    );
    const column<uncompr_f> * const baseCol2 = generate_with_distr(   //create a column with uncompressed values
            countValues,                                              //100M values
            std::uniform_int_distribution<uint64_t>(0, 10),           //uniformly distibuted from 0 to 10
            false                                                     //not sorted
    );
        
    std::cout << "done." << std::endl;

    return 0;
}
~~~
</div>

<h2 id=operators></h2>
Operators
=========

Processing Style
----------------

Before applying any operators, we decide for a vector extension, register size and base type. We call such a combination a processing style.
For the sake of universality, we will use scalar processing on 64 bit registers and unisgned 64-bit integers as base type. It is possible to use a 
different processing style for each operator, but here we will use the same for every operator.
Note that at the moment, only a base type of uint64_t and register sizes according to their vector extension are implemented 
(e.g. avx2 only works with 256-bit registers), but more combinations are in work.


<div class=userCode>
~~~{.cpp}
#include <vector/scalar/extension_scalar.h>
...
    using ve = scalar<v64<uint64_t> >;  //scalar processing with 64-bit registers and 64-bit unsigned integer type
    return 0;
}
~~~
</div>

Select
------

The first operator is a selection on baseCol1, where every value matching <i>150</i> is a hit. The according indexes are stored 
in another column <i>i1</i>.

<div class=userCode>
~~~{.cpp}
#include <core/operators/general_vectorized/select_uncompr.h>
...
    std::cout << "Query execution started... ";
    std::cout.flush();
    
    // Positions fulfilling "baseCol1 = 150"
    auto i1 = morphstore::select<      //explicit namespace is necessary due to another function in c++ called select
            equal,                     //which type of comparison do we want to use?
            ve,                        //The processing style mentioned above
            uncompr_f,                 //output column (i1) is uncompressed
            uncompr_f                  //input (baseCol1) column is uncompressed
    >(baseCol1, 150);                  //column and predicate
    return 0;
}
~~~
</div>

Note that there are also other headers implementing the same operators. In core/scalar and core/vectorized, there are simple 
hand implemented operators, which we use as a base line. However, it is not recommended using them in queries since some
of them have a different interface and they only support scalar and avx2 processing styles.
If the data should be compressed, the headers core/operators/general_vectorized/<operator>_compr.h should be used. They also work 
with uncompressed data.

Project
-------

The second operator in our query is a project. The intermediate result is stored in a new column <i>i2</i>.

<div class=userCode>
~~~{.cpp}
#include <core/operators/general_vectorized/project_uncompr.h>
...
    // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto i2 = project<
              ve,               //The processing style mentioned above
              uncompr_f,        //output column (i2) is uncompressed
              uncompr_f,        //input column (baseCol2) is uncompressed
              uncompr_f>        //input position column (i2) is uncompressed
              (baseCol2, i1);   //column and positions
              
    return 0;
}
~~~
</div>

Aggregation
-----------

The third and last operator is an aggregation, which sums up all values in the intermediate result <i>i2</i>. The final result is
stored in a new column <i>i3</i>.

<div class=userCode>
~~~{.cpp}
#include <core/operators/general_vectorized/agg_sum_uncompr.h>
...
    auto i3 = agg_sum<
                    ve,          //The processing style mentioned above
                    uncompr_f>   //input column (i2) is uncompressed
                    (i2); 
                    
    std::cout << "done." << std::endl << std::endl;                    
    return 0;
}
~~~
</div>
             
Show the result
---------------

There is a convenience function to print the content of a column

<div class=userCode>
~~~{.cpp}
#include <core/utils/printing.h>
...
    print_columns(
          print_buffer_base::decimal,      //print numbers
          i3,                              //column i3
          "SUM(baseCol2)");                //a name for column i3
    return 0;
}
~~~
</div>    

             
Alternative includes
--------------------

Instead of including each individual header for the processing style, all required processing styles can be included at once. This is
useful if more than one processing style is used in a query.

<div class=userCode>
~~~{.cpp}
#include <vector/vector_extension_structs.h>
~~~
</div>

<h2 id=compile></h2>
Compile and Run
===============

The easiest way to test the finished query is to modify the file CMakeLists.txt in src/examples. At the end of the file add
the following code and replace <my_file>.cpp with the name of your file. Note that this only works if your source file is 
also in src/examples.

<div class=userCode>
~~~{cmake}
if ( CTEST_ALL OR CTEST_QUERIES )
add_executable( hello_world_query <my_file>.cpp )
target_compile_options( hello_world_query PRIVATE
                      #any special compile options needed?
                      #should only be necessary if you have some kind of freaky exotic system
                       )
target_link_libraries( hello_world_query PRIVATE "-ldl" )
add_test( hello_world_query_test hello_world_query )
endif(CTEST_ALL OR CTEST_QUERIES)
~~~
</div>

Now the build.sh script in the root folder can be called with the -tQ flag, eg.:

~~~{.sh}
./build.sh -deb -tQ
~~~

If everything works, the following lines should show after a few seconds:

<pre>
    Start 1: example_query_test
1/3 Test #1: example_query_test ...............   Passed    2.64 sec
    Start 2: select_sum_query_test
2/3 Test #2: select_sum_query_test ............   Passed    8.47 sec
    Start 3: hello_world_query_test
3/3 Test #3: hello_world_query_test ...........   Passed    8.46 sec

100% tests passed, 0 tests failed out of 3

Total Test time (real) =  19.57 sec
</pre>

The binary hello_world_query has been written to build/src/examples. If it is executed directly via ./hello_world_query,
the following lines should be prompted:

<pre>
Base data generation started... done.
Query execution started... done.

index  SUM(baseCol2)
    0  .............5003687
    1  (end)
</pre>
