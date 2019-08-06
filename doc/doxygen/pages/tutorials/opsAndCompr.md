\page operators Available Operators and Compressions

Operators
=========

Database Operators
------------------

MorphsStore implements its database operators in different ways:

- Hand implemented (include/core/scalar and include/core/vectorized): Very basic scalar and AVX2 implementations for uncompressed 64-bit integer data, which serve as a baseline. Because of the limited customizability and the slightly different interface, it is not recommended to use them for any real world scenario.
- VectorLib-enabled for uncompressed data (include/core/general_vectprized/*uncompr.h): Operators for uncompressed data using the vector lib, i.e. these operators work on all processing styles, which have a fully implemented set of primitives. The only restriction is that the data must be uncompressed. The output is also uncompressed.
- VectorLib-enabled for compressed data (include/core/general_vectprized/*compr.h): Like above, but also works on compressed data.

The available operators are:
- Select (equal, less, lessequal, greater, greater equal)
- Project
- Calc unary (invert aka. change the sign of a number)
- Calc binary (add, subtract, multiply, divide, modulo)
- Aggregate (Sum)
- Group
- Intersect on sorted sets
- Merge on sorted sets
- Join

The usage of a select for uncompressed data is shown in the following example. Initially, a test data set with 100 sequential numbers is created. 
Then a select is executed twice, once with AVX2 on 256-bit registers, and once with SSE on 128 bit registers. Finally, a simple <i>memcmp</i> tests if the result set from both versions is the same.

<div class=userCode>  
main.cpp   
~~~{.cpp} 

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <iostream>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/operators/general_vectorized/select_uncompr.h>

#define TEST_DATA_COUNT 100

int main(){  
  
  using namespace vectorlib;
  using namespace morphstore;

  std::cout << "Generating..." << std::flush;
  const column< uncompr_f > * testDataColumnSorted = generate_sorted_unique(TEST_DATA_COUNT,0,1);
   
  std::cout << "Done...\n";


  auto result = morphstore::select<greater, avx2<v256<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, 10 );
  auto result1 = morphstore::select<greater, sse<v128<uint64_t>>, uncompr_f, uncompr_f>( testDataColumnSorted, 10 );

  const bool allGood =
     memcmp(result->get_data(),result1->get_data(),result1->get_count_values()*8);

  return allGood;

}
~~~
</div> 

All operator calls follow the same structure: __operator<optional operation, processing style, output format(s), input format(s)>(input column(s), optional predicate, optional selectivity estimation)__.
The calls of the available operators are:

<div class=morphStoreBaseCode>
Select
~~~{.cpp}
const column<t_out_pos_f> * =
select<
        template<typename> class t_op,                  //equal, less, greater, lessequal, or greaterequal      
        class t_vector_extension,                       //e.g. sse<v128<uint64_t>>
        class t_out_pos_f,                              //format of output, e.g. uncompr_f
        class t_in_data_f >(                            //format of input, e.g. uncompr_f
        const column<t_in_data_f> * const inDataCol,    //input column
        const uint64_t val,                             //predicate
        const size_t outPosCountEstimate = 0  );
~~~
</div>


<div class=morphStoreBaseCode>
Project
~~~{.cpp}
const column<t_out_data_f> * =
project<
        class t_vector_extension,                     //e.g. sse<v128<uint64_t>>
        class t_out_data_f,                           //format of output, e.g. uncompr_f
        class t_in_data_f,                            //format of input, e.g. uncompr_f
        class t_in_pos_f >(                           //format of position column, e.g. uncompr_f
        const column<t_in_data_f> * const inDataCol,  //input column
        const column<t_in_pos_f> * const inPosCol );   //position column
~~~
</div>

<div class=morphStoreBaseCode>
Aggregate
~~~{.cpp}
column<uncompr_f> const *  =
agg_sum<
   class VectorExtension,                           //e.g. sse<v128<uint64_t>>
   class t_in_data_f >(                             //format of input, e.g. uncompr_f
   const column<InFormatCol> * const inDataCol );   //input column, the output of an aggregation is uncompressed
~~~
</div>


<div class=morphStoreBaseCode>
Calc
~~~{.cpp}
const column<t_out_data_f> * =
calc_unary<
        template<typename> class t_binary_op,                 //e.g. inv
        class t_vector_extension,                             //e.g. sse<v128<uint64_t>>
        class t_out_data_f,                                   //format of output
        class t_in_data_f >(                                  //format of input, e.g. uncompr_f
        const column<t_in_data_f> * const inDataLCol)
        
const column<t_out_data_f> * =
calc_binary<
        template<typename> class t_binary_op,                 //e.g. add
        class t_vector_extension,                             //e.g. sse<v128<uint64_t>>
        class t_out_data_f,                                   //format of output, e.g. uncompr_f
        class t_in_data_l_f,                                  //format of first input column
        class t_in_data_r_f >(                                //format of second input column
        const column<t_in_data_l_f> * const inDataLCol,       //first input column
        const column<t_in_data_r_f> * const inDataRCol);      //second input column  
~~~
</div>

<div class=morphStoreBaseCode>
Group
~~~{.cpp}
const std::tuple<
        const column<t_out_gr_f> *,                       //first output column contains the group IDs
        const column<t_out_ext_f> * > =                   //the second output column contains the groups
group<
        class t_vector_extension,                         //e.g. sse<v128<uint64_t>>
        class t_out_gr_f,                                 //format of group IDs, e.g. uncompr_f
        class t_out_ext_f,                                //format of groups
        class t_in_data_f>(                               //format of input column
        const column<t_in_data_f> * const inDataCol,
        const size_t outExtCountEstimate = 0);

        
const std::tuple<
        const column<t_out_gr_f> *,                       //first output column contains the group IDs
        const column<t_out_ext_f> * > =                   //the second output column contains the groups
group<
        class t_vector_extension,                         //e.g. sse<v128<uint64_t>>
        class t_out_gr_f,                                 //format of group IDs, e.g. uncompr_f
        class t_out_ext_f,                                //format of groups
        class t_in_gr_f,                                  //format of group IDs from previous group
        class t_in_data_f>(                               //format of input column
        const column<t_in_gr_f> * const inGrCol,          //A column of group IDs obtained by a previous grouping step. Must contain as many data elements as inDataCol.
        const column<t_in_data_f> * const inDataCol,      //input column
        const size_t outExtCountEstimate = 0 );           //optional selectivity estimation
~~~
</div>

<div class=morphStoreBaseCode>
Merge
~~~{.cpp}
const column<t_out_pos_f> * =
merge_sorted<
            class t_vector_extension,                       //e.g. sse<v128<uint64_t>>
            class t_out_pos_f,                              //format of output
            class t_in_pos_l_f,                             //format of first input column
            class t_in_pos_r_f >(                           //format of second input column
            const column<t_in_pos_l_f> * const inPosLCol,   //first input column
            const column<t_in_pos_r_f> * const inPosRCol);  //second input column
~~~
</div>

<div class=morphStoreBaseCode>
Join
~~~{.cpp}
std::tuple<
   column< OutFormatLCol > const *,
   column< OutFormatRCol > const *
> const =
join<
   class VectorExtension,                               //e.g. sse<v128<uint64_t>>
   class OutFormatLCol,                                 //format of first output column
   class OutFormatRCol,                                 //format of second output column
   class InFormatLCol,                                  //format of first input column
   class InFormatRCol >(                                //format of second input column
   column< InFormatLCol > const * const p_InDataLCol,   //first input column
   column< InFormatRCol > const * const p_InDataRCol,
   size_t const outCountEstimate = 0 );

column<OutFormatCol> const *  =
semi_join<
   class VectorExtension,                               //e.g. sse<v128<uint64_t>>
   class OutFormatCol,                                  //format of output
   class InFormatLCol,                                  //format of first input column
   class InFormatRCol >(                                //format of second input column
   column< InFormatLCol > const * const p_InDataLCol,   //first input column
   column< InFormatRCol > const * const p_InDataRCol,   //second input column
   size_t const outCountEstimate = 0 );                 //optional selectivity estimation
~~~
</div>

<div class=morphStoreBaseCode>
Intersect
~~~{.cpp}
const column<t_out_pos_f> * =
intersect_sorted<
        class t_vector_extension,                       //e.g. sse<v128<uint64_t>>
        class t_out_pos_f,                              //format of output
        class t_in_pos_l_f,                             //format of first input column
        class t_in_pos_r_f >(                           //format of second input column
        const column<t_in_pos_l_f> * const inPosLCol,   //first input column
        const column<t_in_pos_r_f> * const inPosRCol,   //second input column
        const size_t outPosCountEstimate = 0 );         //optional selectivity estimation
~~~
</div>

Note that currently, some operators for compressed data do have different names, i.e. my_select_wit_t and my_project_wit_t. 


Special Operators
-----------------

### The Morph Operator

The operators in include/core/general_vectprized/*compr.h can work with differently compressed data. If the input format of the operator does not equal the output format of its predecessor, the formats have to be changed.
This is what the morph operator is for. The only parameter it takes, is a column. The processing style and destination format are passed as template arguments. For instance, the following code transforms a given column into 
a column, which is compressed with dynamic bitpacking using AVX512:

<div class=morphStoreBaseCode>
Q1.1
~~~{.cpp}
...
auto date_d_year__d = morph<avx512<v512<uint64_t>>, dynamic_vbp_f<512, 32, 8>>(date.d_year);
...
~~~
</div>

Compression
===========
<div class=ToDo> </div>