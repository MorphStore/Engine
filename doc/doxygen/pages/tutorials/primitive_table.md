\page primitiveTable Available Primitives

IO
--------

| bitwidth base type            | Extension     | Vector size (Bit) | Primitives               |                                     |    | | |  | | |
| -----------                   | ------------- | ----------------- | ---------------          | -------------                       |                            |                           |                                   |                           |                       |                       |
|                               |               |                   | load unaligned           |  load aligned                       | gather (unaligned)         | store unaligned           | store aligned                     | compressstore (unaligned) |  stream load          | steam store           |
| 64                            | None (scalar) |  64               | <div class=YES>int</div> |  <div class=YES>int, double</div>   |   <div class=YES>int</div> |  <div class=YES>int</div> |  <div class=YES>int, double</div> |  <div class=YES>int</div> |  <div class=NO></div> | <div class=NO></div>            | 
|                               | SSE           | 128               | <div class=YES>int *</div> |  <div class=YES>int, double</div>   |   <div class=YES>int</div> |  <div class=YES>int</div> |  <div class=YES>int, double</div> |  <div class=YES>int</div> |  <div class=YES>int</div> | <div class=YES>int</div>  |
|                               | AVX2          | 128               | <div class=NO></div>     | <div class=NO></div>                | <div class=YES>int</div>   |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=YES>int *</div> |  <div class=YES>int, double</div>   |   <div class=YES>int</div> |  <div class=YES>int</div> |  <div class=YES>int, double</div> |  <div class=YES>int</div> |  <div class=YES>int</div> | <div class=YES>int</div>  |
|                               | AVX512        | 128               | <div class=NO></div> | <div class=NO></div>         | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=YES>int</div>      |  <div class=NO></div> | <div class=NO></div>  |
|                               | AVX512        | 256               | <div class=NO></div> | <div class=NO></div>         | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=YES>int</div>      |  <div class=NO></div> | <div class=NO></div> |
|                               | AVX512        | 512               | <div class=YES>int</div> |  <div class=YES>int, double</div>   |   <div class=YES>int</div> |  <div class=YES>int</div> |  <div class=YES>int, double</div> |  <div class=YES>int</div> |  <div class=YES>int</div> | <div class=YES>int</div>  |
| 32                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=YES>float</div> |  <div class=YES>float</div>        |    <div class=NO></div>    |  <div class=YES>float</div>|  <div class=YES>float</div>     |  <div class=NO></div>     |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=YES>float</div> |  <div class=YES>float</div>        |    <div class=NO></div>    |  <div class=YES>float</div>|  <div class=YES>float</div>     |  <div class=NO></div>     |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX512        | 512               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
| 16                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | SSE           | 128               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX2          | 256               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX512        | 512               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|  8                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | SSE           | 128               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX2          | 256               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>      | <div class=NO></div>       |
|                               | AVX512        | 512               |  <div class=HACKY>**</div> | <div class=HACKY>**</div>          | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=HACKY>**</div> | <div class=HACKY>**</div>  |

<p> \* There is an additional unaligned load with the IO version UNALIGNEDX, which uses a different intrinsic than the normal unaligned load. </p>
<p> \*\* According 64-bit primitive can be used if input/output address is casted to a 64-bit integer pointer. </p>
 
 
Create
----------

| bitwidth base type            | Extension     | Vector size (Bit) | Primitives                 |                                    |                       |
| -----------                   | ------------- | ----------------- | ---------------            | -------------                      | ----------------      |
|                               |               |                   | set1                       |  set sequence                      |  set                  |
| 64                            | None (scalar) |  64               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=NO></div>  |
|                               | SSE           | 128               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=YES>int</div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX2          | 256               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=YES>int</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 512               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=YES>int</div>  |
| 32                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | SSE           | 128               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=YES>int</div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX2          | 256               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=YES>int</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 512               | <div class=YES>int</div>   | <div class=YES>int</div>           | <div class=YES>int</div>  |
| 16                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|  8                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>  |


Extract
------------

| bitwidth base type            | Extension     | Vector size (Bit) | Primitives                 |
| -----------                   | ------------- | ----------------- | ---------------            |
|                               |               |                   | extract value              |
| 64                            | None (scalar) |  64               | <div class=YES>int</div>   |
|                               | SSE           | 128               | <div class=YES>int</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=YES>int</div>   |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=YES>int</div>   |
| 32                            | None (scalar) |  64               | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=YES>int</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=YES>int</div>   |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=YES>int</div>   |
| 16                            | None (scalar) |  64               | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=NO></div>       |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=NO></div>       |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=NO></div>       |
|  8                            | None (scalar) |  64               | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=NO></div>       |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=NO></div>       |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=NO></div>       |




Comparisons
-------------

| bitwidth base type            | Extension     | Vector size (Bit) | Primitives                 |                                    |                            |                          |                                   |                           |
| -----------                   | ------------- | ----------------- | ---------------            | -------------                      |                            |                          |                                   |                           |
|                               |               |                   | equal                      |  less                              | less equal                 | greater                  | greater equal                     | count matches             |
| 64                            | None (scalar) |  64               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> | 
|                               | SSE           | 128               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |
| 32                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
| 16                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|  8                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |

Calculation
--------------


| bitwidth base type            | Extension     | Vector size (Bit) | Primitives                 |                                    |                            |                          |                                   |                           |                 |                 |                         |                 |                           |
| -----------                   | ------------- | ----------------- | ---------------            | -------------                      |                            |                          |                                   |                           |                 |                 |                         |                 |                           |
|                               |               |                   | add                        |  subtract                          | horizontal add             | multiply                 | divide                            | modulo                    | invert                      | shift left               | shift left (individual)           | shift right               | shift right (individual)  |
| 64                            | None (scalar) |  64               | <div class=YES>int</div>   |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div> |
|                               | SSE           | 128               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div> |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div> |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |  <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=YES></div>      |  <div class=YES>int</div>          | <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div>   | <div class=YES>int</div> |  <div class=YES>int</div>         |  <div class=YES>int</div> |  <div class=YES>int</div> |
| 32                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
| 16                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|  8                            | None (scalar) |  64               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | SSE           | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX2          | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |
|                               | AVX512        | 512               | <div class=NO></div>       | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      | <div class=NO></div>               | <div class=NO></div>       |  <div class=NO></div>    |  <div class=NO></div>             | <div class=NO></div>      |


Logic
------------

| bitwidth base type            | Extension     | Vector size (Bit) | Primitives                 |                      |
| -----------                   | ------------- | ----------------- | ---------------            |                      |
|                               |               |                   | bitwise AND                | bitwise OR           |
| 64                            | None (scalar) |  64               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
|                               | SSE           | 128               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX2          | 256               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 512               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
| 32                            | None (scalar) |  64               | <div class=HACKY>*</div>   | <div class=HACKY>*</div>   |
|                               | SSE           | 128               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX2          | 256               | <div class=YES>int, other*</div> | <div class=YES>int, other*</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 512               | <div class=YES>int, other*</div> | <div class=YES>int, other*</div>  |
| 16                            | None (scalar) |  64               |  <div class=HACKY>*</div>  | <div class=HACKY>*</div>   |
|                               | SSE           | 128               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX2          | 256               | <div class=YES>int, other*</div> | <div class=YES>int, other*</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 512               | <div class=YES>int, other*</div> | <div class=YES>int, other*</div>  |
|  8                            | None (scalar) |  64               | <div class=HACKY>*</div>   | <div class=HACKY>*</div>   |
|                               | SSE           | 128               | <div class=YES>int, other*</div>| <div class=YES>int, other*</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX2          | 256               | <div class=YES>int, other*</div> | <div class=YES>int, other*</div>  |
|                               | AVX512        | 128               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 256               | <div class=NO></div>       | <div class=NO></div>                    |
|                               | AVX512        | 512               | <div class=YES>int, other*</div> | <div class=YES>int, other*</div>  |

<p> \* Primitive can be used when data is casted to a 64-bit integer (vector). </p>


Manipulate
-------------

| bitwidth base type            | Extension     | Vector size (Bit) | Primitives                 |
| -----------                   | ------------- | ----------------- | ---------------            |
|                               |               |                   | rotate              |
| 64                            | None (scalar) |  64               | <div class=YES>int</div>   |
|                               | SSE           | 128               | <div class=YES>int</div>   |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=YES>int</div>   |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=YES>int</div>   |
| 32                            | None (scalar) |  64               | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=NO></div>       |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=NO></div>       |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=NO></div>       |
| 16                            | None (scalar) |  64               | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=NO></div>       |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=NO></div>       |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=NO></div>       |
|  8                            | None (scalar) |  64               | <div class=NO></div>       |
|                               | SSE           | 128               | <div class=NO></div>       |
|                               | AVX2          | 128               | <div class=NO></div>       |
|                               | AVX2          | 256               | <div class=NO></div>       |
|                               | AVX512        | 128               | <div class=NO></div>       |
|                               | AVX512        | 256               | <div class=NO></div>       |
|                               | AVX512        | 512               | <div class=NO></div>       |

