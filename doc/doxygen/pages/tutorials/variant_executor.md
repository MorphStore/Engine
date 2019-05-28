Testing and µ-Benchmarking Operator-Variants
=====

Motivation
-----

In MorphStore, we usually have multiple variants of one operator, since we consider different processing styles, or vector extensions, as well as different compressed formats.
It is crucial for us to be able to compare these variants with respect to:

- correctness, i.e., whether they all produce the same output if given the same input
- performance, i.e., how long the execution takes

MorphStore has a utility for these purposes: `variant_executor`.

`variant_executor` Overview
-----

`variant_executor` enables the comparison of multiple operator variants for multiple combinations of input parameters.
A combination of the input parameters is called *setting* hereafter.

### Outputs

Two kinds of outputs are generated:

- to `stdout`
  - status/progress outputs, including the results of correctness checks
  - human readable format
- to `stderr`
  - detailled monitoring data
  - tab-separated format
  - MorphStore must be compiled with `-mon` to use this feature
  
Due to the separation between `stdout` and `stderr`, any of the two can easily be redirected to a file or to `/dev/null`.

### General Usage Notes

Internally, `variant_executor` is implemented in a very generic way.
As a consequence, it must be specialized a little to use it.

### Limitations

- `variant_executor` requires the ability to free columns.
Thus, at the time of this writing, MorphStore should be compiled with `-noSelfManaging` to use it.
- At the moment, it is ready to be used for testing. Using it for reliable micro-benchmarks, however, will still need some more work.

Usage
-----

This section explains the basics of using `variant_executor` and illustrates it with the example of the project-operator.
**The complete and compilable source code of this example can be found in `src/examples/variant_executor_usage.cpp`.**
The executable is `build/src/examples/variant_executor_usage`.

### Including the Right Header

<div class=userCode>
~~~{.cpp}
	#include <core/utils/variant_executor.h>
~~~
</div>

### Creating an Instance

First, select the right template-specialization of `variant_executor`.
We must specialize for:

- the interface of the operator to test/benchmark
  - this is fully determined by the operator, so there is nothing to choose here for the developer
  - this includes
    - the number of output columns
    - the number of input columns
    - the types of additional non-column parameters (such as a selection constant or a cardinality estimate), if applicable for the operator
- things specific to the test/benchmark to be implemented
  - this is completely up to the developer of the test/benchmark
  - this includes
    - the types of the *variant parameters*, i.e., the values that identify a variant for this test/benchmark
    - the types of the *setting parameters*, i.e., the values that identify a setting for this test/experiment
  
Example: The project operator has one output column, two input columns, and no additional parameters.
Furthermore, let us assume we want to evaluate the different processing style variants of the project-operator (but no compressed formats etc.).
Then, a single `std::string` suffices for the variant parameters.
Note that we could also use `morphstore::processing_style_t` as the type of this variant parameter, but for the output to the console, the type must be insertable into a stream.
Interesting setting parameters could be the number of data elements in the project-operator's input data column and input positions column.
Hence, the type of `variant_executor` we need is:

<div class=userCode>
~~~{.cpp}
	using varex_t = variant_executor_helper<1, 2>::type
		::for_variant_params<std::string>
		::for_setting_params<size_t, size_t>;
~~~
</div>

If our operator haf additional non-column parameters, then their types would follow behind the numbers of output and input columns, e.g., `<1, 2, size_t>`.
It is recommendable to use the alias name `varex_t` here, since this type will be needed again in the following.

Next, we need an instance.
The constructor expects the *names* of the variant parameters, the setting parameters, and the operator's additional non-column parameters as vectors of strings.

<div class=userCode>
~~~{.cpp}
	varex_t varex(
	        {"ps"}, // names of the variant parameters
	        {"inDataCount", "inPosCount"}, // names of the setting parameters
  	        {} // names of the operator's additional parameters
  	);
~~~
</div>

### Creating Variants

Essentially, a variant is just a function pointer.
However, in practice two requirements make the case a little more complex:

1. We need to identify the variant somehow for the console output.
A function address would not be suitable for that.
Therefore, a variant is actually a `std::tuple`, including also values for the variant parameters mentioned above.
2. We also consider operators on compressed data.
In fact, the variants of *one* operator for *different* compressed formats have different interfaces, e.g., they require columns of different formats.
Consequently, there is no common function pointer type for these variants.
Therefore, instead of a function pointer, we have to provide a `varex_t::operator_wrapper`, which handles these interface mismatches internally.

The precise type of a variant for our specialization of `variant_executor` can be accessed using the alias `varex_t::variant_t`.
For instance, the scalar uncompressed variant of the project-operator in our case could be created as follows:

<div class=userCode>
~~~{.cpp}
	// This is a std::tuple.
	varex_t::variant_t myVariant = {
	        // Wrapper for the function pointer.
	         new varex_t::operator_wrapper::for_output_formats<uncompr_f>::for_input_formats<uncompr_f, uncompr_f>(
	                // Function pointer to the actual operator function.
	                &project<processing_style_t::scalar, uncompr_f, uncompr_f, uncompr_f> 
	        ),
	        // The variant key.
	        "scalar"
	};
~~~
</div>

First, we take the function pointer.
Here, we have to specify all necessary template parameters to cleary identify the variant we want.
Then, we wrap the function pointer into a `varex_t::operator_wrapper`.
Unfortunately, with the current implementation we have to repeat the output and input formats used by the operator variant.
Finally, we add the variant parameters, in this case `"scalar"`.

In fact, many things in the above definition of a variant depend on each other.
For instance, the variant parameter `"scalar"` should always match the processing style that is actually used in the function pointer.
Thus, it is recommendable to define a macro that expands to an initializer list for a variant.
In our example, this macro should have the processing style as its only argument:

<div class=userCode>
~~~{.cpp}
	#define MAKE_VARIANT(ps) \
	{ \
	    new varex_t::operator_wrapper::for_output_fo rmats<uncompr_f>::for_input_formats<uncompr_f, uncompr_f>( \
	            &project<processing_style_t::ps, uncompr_f, uncompr_f, uncompr_f> \
	    ), \
	    STR_EVAL_MACROS(ps) \
	}
~~~
</div>

Now it is quite easy to define the three variants we want to execute:

<div class=userCode>
~~~{.cpp}
	const std::vector<varex_t::variant_t> variants = {
	    MAKE_VARIANT(scalar),
	    MAKE_VARIANT(vec128),
	    MAKE_VARIANT(vec256)
	};
~~~
</div>

### Creating Settings

As mentioned above, a setting is a combination of parameters/inputs for the operator variants.
Since all operators consume at least one column (and maybe more columns or non-column inputs), we have to provide such columns for our test/benchmark.
Since the very exact data does usually not matter, we introduce the abstration of *setting parameters*.
The setting parameters are meant to be some *simple* values which fully specify a setting for our case and which are also suitable for the console output.
For instance, for the project-operator, the number of data elements in the input data column and input positions column could be interesting.
`varex_t::setting_t` is the type alias of a combination of setting parameters.
In fact, this is simply a `std::tuple` of the types we provided in `for_setting_params` above, i.e., `<size_t, size_t>`.

<div class=userCode>
~~~{.cpp}
	// Define the setting parameters.
	const std::vector<varex_t::setting_t> settingParams = {
	    // inDataCount, inPosCount
	    {100, 1000},
	    {123, 1234}
	};
~~~
</div>

### Execute Variants for Settings

Now we come to the actual execution of the variants in the different settings.
Here, we iterate over the combinations of setting parameters we have just defined.

<div class=userCode>
~~~{.cpp}
	for(const varex_t::setting_t sp : settingParams) {
	    // Extract the individual setting parameters.
	    size_t inDataCount;
	    size_t inPosCount;
	    std::tie(inDataCount, inPosCount) = sp;
	    
	    // ...
	}
~~~
</div>

For each such combination, we must do the following:

Generate the data.
Note that we always create *uncompressed* data, even if some of the variants need compressed inputs, since `variant_executor` handles such cases internally.
It is recommended to call `varex.print_datagen_started()` and `varex.print_datagen_done()` before and after the data generation, repectively.
These functions produce some progress output informing the user about the ongoing data generation.
In the following, note how the genration of `inPosCol` depends on the number of data elements of both the input data column and the input position column.

<div class=userCode>
~~~{.cpp}
	// Generate the data.
	varex.print_datagen_started();
	auto inDataCol = generate_with_distr(
	    inDataCount,
	    std::uniform_int_distribution<uint64_t>(100, 200),
	    false
	);
	auto inPosCol = generate_with_distr(
	    inPosCount,
	    std::uniform_int_distribution<uint64_t>(0, inDataCount - 1),
	    false
	);
	varex.print_datagen_done();
~~~
</div>

Execute the variants for the current setting by calling `varex.execute_variants()` with the variants, the setting parameters, and the setting itself (i.e., the input columns and additional parameters, if applicable for the operator).
This takes care of a lot of important things such as morphing the uncompressed input columns to the formats needed by the variants, executing the variants, calling the monitoring appropriately, morphing their outputs back to uncompressed data, comparing the outputs of different variants, and freeing all columns it created internally.

<div class=userCode>
~~~{.cpp}
	// Execute the variants.
	varex.execute_variants(
	    // Variants to execute
	    variants,
	    // Setting parameters
	    inDataCount, inPosCount,
	    // Input columns / setting
	    inDataCol, inPosCol
	);
~~~
</div>

In the end, we must delete the generated columns. Note that `variant_executor` ensures that all columns that were created *internally* during the variant execution are actually freed.

<div class=userCode>
~~~{.cpp}
	// Delete the generated data.
	delete inPosCol;
	delete inDataCol;
~~~
</div>

### Finally

In the end, the following two methods are useful:

- `varex.done()`: Prints a summary of the checks and -- if MorphStore was compiled with `-mon` -- the monitoring data.
- `varex.good()`: Returns a boolean indicating whether *all* checks were successful.

Interpreting the Outputs
-----

### Progress/status information on `stderr`

This output is printed as things happen during the execution.
It indicates

- when data generation starts and ends
- for each setting, the settings parameters
- for each variant, the start and end, as well as the check result
- whether all checks were okay in the final summary

The output of our example looks like this:

	Data generation: started... done.
	Setting
		Parameters
			inDataCount: 	100
			inPosCount: 	1000
		Executing Variants
			ps	
			scalar	: started... done. -> reference
			vec128	: started... done. -> ok
			vec256	: started... done. -> ok
	Data generation: started... done.
	Setting
		Parameters
			inDataCount: 	123
			inPosCount: 	1234
		Executing Variants
			ps	
			scalar	: started... done. -> reference
			vec128	: started... done. -> ok
			vec256	: started... done. -> ok
	Summary
		all ok

### Monitoring data on `stdout`

This output is printed only if MorphStore was compiled with `-mon`.
It is printed in the end, when `varex.done()` is called.
The output is a tab-separated table.
Its columns are (from left to right):

- the variant parameters we specified
- the setting parameters we specified
- a column for the runtime (in microseconds) of the operator variant
- a column for the correctness check of the operator variant
  - `-1` (reference): this variant produced the reference output
  - `0` (not ok): this variant did *not* produce the same output as the reference
  - `1` (ok): this variant did produce the same output as the reference

The output of our example looks like this:

	LogFilename: 2019-05-28-17:15:13_monitoringLog
	JSonLogFilename: 2019-05-28-17:15:13_monitoringLog
	ps	inDataCount	inPosCount	runtime:µs	check
	scalar	100	1000	3	-1
	vec128	100	1000	2	1
	vec256	100	1000	6	1
	scalar	123	1234	9	-1
	vec128	123	1234	6	1
	vec256	123	1234	10	1

Summary
-----

`variant_executor` is, hopefully, a useful utility for creating tests and micro-benchmarks for the multitude of operator variants we have in MorphStore.
It hides a lot of the complex things involved in such tests/benchmarks from the developer.
Nevertheless, some specialization and tailoring to the particular case is still required.
However, this should be close to a minimum and usually follows the same pattern.
