@page scalableVectorLengths Support of Scalable Vector Lengths

SX Aurora Tsubasa is a dedicated vector processor in the shape of a PCIe card.
It has 8 vector cores, 24 GB of HBM2 memory featuring a theoretical memory bandwidth of 
1 TB/s and a peak performance up to 2.15 TFLOPS (double precision).
There are intrinsics that allow comfortable use of the vector instructions.
Each register of SX Aurora Tsubasa can hold 16.384 Bit.
Therefore parallel processing of 256 64-Bit elements or 512 32-bit elements is possible.
However, the number of elements that is processed aka the length of the vector can be chosen
by the programmer.
The following page describes the integration of such feature -scalable vector lengths - into MorphStore and TVL.

Vectorized Processing in MorphStore
-----------------------------------

![Processing](processing.png)

<p style="text-align: center;">Figure 1: Processing in MorphStore</p>

The figure above shows the processing of a column in MorphStore.
The underlying processing model is Operator at a time (OaaT).
A column possibly consists of two parts: 
One compressed part and one uncompressed part.
First of all the column is divided by the operator:
<ul>
<li>The compressed part is processed vectorized</li>
<li>As much as possible of the uncompressed part is processed vectorized as well.</li>
<li>The rest of the uncompressed part that does not fit an entire vector register is processed. 
This is because scalable vector lengths are not supported by most of SIMD extensions.</li>
</ul>
For every part of the column the decompress_and_process_batch is called.
This batch calls the core operator (processing unit) until the necessary amount of elements 
was processed.
To do so primitives from the TVL are utilized.
Depending on the operator the result is written back either in a non selective or selective manner.

Conceptual Integration of Scalable Vector Lengths
-------------------------------------------------

Inside TVL specializations of primitives for SX Aurora had to be implemented.
Also the interface of the primitives had to be extended in order to support a parameter
element_count used to pass the amount of elements to be processed by the primitive.
Vector extension structs gained a new property is_scalable. This indicates whether the extension
support scalable vector lengths or not.

|              | Extension | is_scalable |
| ------------ | --------- | ------------------- |
| Intel (AMD)  | MMX       | false               |
|              | SSE       | false               |
| ARM          | NEON v1   | false               |
|              | NEON v2   | false               |
| Intel (AMD)  | AVX(2)    | false               |
|              | AVX512    | false               |
| TSUBASA (NEC)| TSUBASA   | true                |
<p style="text-align: center;">Table 1: Support of scalable vector lengths by SIMD extension</p>


In MorphStore the methods calling primitives had to be overloaded or specialized to support element_count parameter.
This includes methods of the core operator (processing unit), the operator, the methods to read and write.
The decompress_and_process_batch had to be specialized as well.
Inside the operators a state is used to accumulate the result of all 3 calls of decompress_and_process_batch.
The use of this state has to be adjusted a little when the 3rd call of the batch is not scalar but vectorized.

Processing of a column is quite similar as before.
The column is still divided in 3 parts.
The first one is processed vectorized and compressed.
The second one is uncompressed and is processed vectorized.
The third part is uncompressed, smaller than one vector register and yet processed vectorized.
Why not have only 2 parts and 2 calls of decompress_and_process_batch?
Currently the division of a column is done in the operator and the batch only processes the given amounts of elements.
Sticking to this keeps the amount of changes in the operator as well as the batch relatively low.
However, it is completely possible to change this in future.





