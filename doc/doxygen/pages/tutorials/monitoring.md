@page Monitoring

The monitoring framework allows the user to leverage compile-time switches for enabling or disabling software coutners.
Monitoring is enabled, by compiling MorphStore using the "-mon" or "--enable-monitoring" switches.

Currently, only timed intervals and one-time constants can be added.

####Macros
There are a number of predefined macros to allow easy and encapsulated counter access.

<div class=morphStoreDeveloperCode>
~~~{.cpp}
	#define MONITOR_START_INTERVAL( CTR ) 	 morphstore::monitoring::get_instance().startInterval( CTR );
	#define MONITOR_END_INTERVAL( CTR ) 	 morphstore::monitoring::get_instance().stopInterval( CTR );
	#define MONITOR_ADD_PROPERTY( key, val ) morphstore::monitoring::get_instance().addProperty( key, val );
	#define MONITOR_PRINT_COUNTERS( ... ) 	 morphstore::monitoring::get_instance().printCounterData( __VA_ARGS__ );
	#define MONITOR_PRINT_PROPERTIES( ... )	 morphstore::monitoring::get_instance().printPropertyData( __VA_ARGS__ );
	#define MONITOR_PRINT_ALL( ... ) 	 	 morphstore::monitoring::get_instance().printAllData( __VA_ARGS__ );
~~~
</div>

The following code segment visualizes a minimal working example on how to use the monitoring frontend.

<div class=morphStoreDeveloperCode>
~~~{.cpp}
	[...]
	const uint64_t bw = 1000000000;
	MONITOR_ADD_PROPERTY( "firstForLoop", bw )
	MONITOR_START_INTERVAL( "operatorTime" + std::to_string( bw ) )
	for ( size_t i = 0; i < bw; ++i ) {
		vec_o[ i ] = vec_j[ i ] + vec_k[ i ];
	}
	MONITOR_END_INTERVAL( "operatorTime" + std::to_string( bw ) )
	[...]
~~~
</div>


The INTERVAL macros start and stop a respective counter, where CTR is a user defined name. If no counter with the given name currently exists, it will be created.
Calling START_/STOP_INTERVAL multiple times with the same CTR, the newly measured intervall will be recorded an saved as an array of values.

MONITOR_ADD_PROPERTY records a given numerical value for a user-defined string key.

All counters are numbered by the time they get *created*.

Printing counters can be done individually for a given CTR or user-defined string, or for all of them at once.

For printing all counters/properties, there are two parameters for the macro.
The first one is the printing channel, which is either *monitorShellLog* (prints to terminal) or *monitorFileLog* (writes to a file with a default name).
The second one is the selection if either alphanumerical or occurence-based sorting should be applied, default is *false*, i.e. alphanumerical ordering.

<div class=morphStoreDeveloperCode>
~~~{.cpp}
    std::cout << "#### Testing All Counters and Properties not sorted" << std::endl;
    MONITOR_PRINT_ALL( monitorShellLog, true );
    
    std::cout << "#### Testing All Counters not sorted" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog );

    std::cout << "#### Testing All Counters with sorting" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog, true );

    std::cout << "#### Testing All Properties not sorted" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog );

    std::cout << "#### Testing All Properties with sorting" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog, true );

    std::cout << "#### Testing Single Counter" << std::endl;
    MONITOR_PRINT_COUNTERS( monitorShellLog, "operatorTime43" );

    std::cout << "#### Testing Single Property" << std::endl;
    MONITOR_PRINT_PROPERTIES( monitorShellLog, "operatorParam8" );
~~~
</div>


