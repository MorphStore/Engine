/**********************************************************************************************
 * Copyright (C) 2019 by MorphStore-Team                                                      *
 *                                                                                            *
 * This file is part of MorphStore - a compression aware vectorized column store.             *
 *                                                                                            *
 * This program is free software: you can redistribute it and/or modify it under the          *
 * terms of the GNU General Public License as published by the Free Software Foundation,      *
 * either version 3 of the License, or (at your option) any later version.                    *
 *                                                                                            *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;  *
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  *
 * See the GNU General Public License for more details.                                       *
 *                                                                                            *
 * You should have received a copy of the GNU General Public License along with this program. *
 * If not, see <http://www.gnu.org/licenses/>.                                                *
 **********************************************************************************************/

/**
 * @file monitoring.h
 * @brief The monitoring interface helps to get some data out of a benchmark.
 * @todo TODOS?
 */

#ifndef MORPHSTORE_CORE_UTILS_MONITORING_H_
#define MORPHSTORE_CORE_UTILS_MONITORING_H_

#include <map>
#include <unordered_map>
#include <chrono>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <ctime>

#include <core/utils/logger.h>
#include <inttypes.h>

namespace morphstore {

#ifdef MSV_USE_MONITORING
typedef std::chrono::high_resolution_clock::time_point chrono_tp;
typedef std::chrono::high_resolution_clock chronoHighRes;
typedef std::chrono::seconds chronoSec;
typedef std::chrono::milliseconds chronoMilliSec;
typedef std::chrono::microseconds chronoMicroSec;
typedef std::chrono::nanoseconds  chronoNanoSec;

class monitoring_logger {
protected:
	monitoring_logger() {
	}

public:
	virtual ~monitoring_logger() {

	}
	virtual void write( const char* text ) = 0;
	virtual void write( uint64_t val ) = 0;
	virtual void write( double val ) = 0;
};

class monitoring_shell_logger : public monitoring_logger {
private:
	monitoring_shell_logger() {
	}

public:
	void write( const char* text ) override {
		printf( "%s", text );
	}

	void write( uint64_t val ) override {
		printf( "%" PRIu64, val );
	}

	void write( double val ) override {
		printf( "%f", val );
	}

	static monitoring_shell_logger & get_instance(void) {
		static monitoring_shell_logger instance;
		return instance;
	}
};

class monitoring_file_logger : public monitoring_logger {
private:
	std::string* logFileName;

	monitoring_file_logger() {
		auto now = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(now);

		std::stringstream ss;
		ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X_monitoringLog");
		logFileName = new std::string( ss.str().c_str() );
		std::cout << "LogFilename: " << *logFileName << std::endl;
}

public:
	void write( MSV_CXX_ATTRIBUTE_PPUNUSED const char* text ) override {
		error( "[Monitoring] Monitoring File Logger is yet to be implemented -- nothing logged." );
	}
	void write( MSV_CXX_ATTRIBUTE_PPUNUSED uint64_t val ) override {
		error( "[Monitoring] Monitoring File Logger is yet to be implemented -- nothing logged." );
	}

	void write( MSV_CXX_ATTRIBUTE_PPUNUSED double val ) override {
		error( "[Monitoring] Monitoring File Logger is yet to be implemented -- nothing logged." );
	}

	static monitoring_file_logger & get_instance(void) {
		static monitoring_file_logger instance;
		return instance;
	}
};

auto monitorShellLog = morphstore::monitoring_shell_logger::get_instance();
auto monitorFileLog = morphstore::monitoring_file_logger::get_instance();

struct monitoring_parameter {
	const std::string name;
	double value;

	monitoring_parameter( const std::string name, double value ) :
	name( name ),
	value( value )
	{
	}

	void print( monitoring_logger& log ) {
		log.write( name.c_str() );
		log.write( " " );
		log.write( value );
		log.write( "\n" );
	}
};

struct monitoring_counter {
	bool started;
	chrono_tp startTp;
	uint32_t maxValues;
	uint32_t lastValue;
	uint64_t* values;

	const std::string name;

	monitoring_counter( const std::string name ) :
		started( false ),
		maxValues( 1024 ),
		lastValue( 0 ),
		name( name )
	{
		values = (uint64_t*) malloc( maxValues * sizeof( uint64_t ) );
	}

	void start() {
		started = true;
		startTp = chronoHighRes::now();
	}

	void stop() {
		chrono_tp stopTp = chronoHighRes::now();
		started = false;
		values[ lastValue++ ] = static_cast< uint64_t >( std::chrono::duration_cast< chronoMicroSec >( stopTp - startTp ).count() );
		if ( lastValue == maxValues ) {
			maxValues = std::ceil( maxValues * 1.2 );
			debug( "Reallocating with ", maxValues, " values. Ptr is: ", values  );
			uint64_t* newValues = (uint64_t*) realloc( values, maxValues );
			if ( newValues != nullptr ) {
				values = newValues;
			} else {
				throw std::runtime_error("Reallocating of monitoring data array failed.");
			}
		}
	}

	void print( monitoring_logger& logChannel ) {
		std::string valueList ="";
		for ( size_t i = 0; i < lastValue; ++i ) {
			valueList += std::to_string( values[i] );
			if ( i < lastValue - 1 ) {
				valueList += ", ";
			}
		}
		logChannel.write( "[Monitoring] " );
		logChannel.write( name.c_str() );
		logChannel.write( ": " );
		logChannel.write( valueList.c_str() );
		logChannel.write( "\n" );

	}
};

typedef std::map< std::string, monitoring_counter* > monitorCounterMap;
typedef std::map< std::string, monitoring_parameter* > monitorParameterMap;

class monitoring {
public:
    static monitoring & get_instance() {
       static thread_local monitoring instance;
       return instance;
    }

    void startInterval( const std::string& ident ) {
    	monitorCounterMap::iterator counter = counterData.find( ident);
    	if ( counter != counterData.cend() ) {
    		if ( ! counter->second->started ) {
    			counter->second->start();
    		} else {
    			wtf( "[Monitoring] Starting counter with already open interval: " + ident );
    		}
    	} else {
    		monitoring_counter* ct = new monitoring_counter( ident );
    		counterData.insert( {ident, ct} );
    		ct->start();
    	}
    }

    void stopInterval( const std::string& ident ) {
    	monitorCounterMap::iterator counter = counterData.find( ident);
    	if ( counter != counterData.cend() ) {
    		if ( counter->second->started ) {
    			counter->second->stop();
    		} else {
    			wtf( "[Monitoring] Stopping counter which never started: " + ident + " -- No timing taken!" );
    		}
    	}
    }

    void addProperty( const std::string& ident, const double val ) {
    	monitorParameterMap::iterator counter = parameterData.find( ident);
    	if ( counter != parameterData.cend() ) {
    		wtf( "[Monitoring] Trying to add the same parameter twice is not yet supported" );
    	} else {
    		parameterData.insert( { ident, new monitoring_parameter( ident, val ) } );
    	}
    }

    void printCounterData( monitoring_logger& log ) {
    	for ( auto counter : counterData ) {
    		counter.second->print( log );
    	}
    }

    void printCounterData( monitoring_logger& log, const std::string& ident ) {
    	monitorCounterMap::iterator counter = counterData.find( ident);
    	if ( counter != counterData.cend() ) {
    		counter->second->print( log );
    	}
    }

    void printPropertyData( monitoring_logger& log ) {
    	for ( auto property : parameterData ) {
    		property.second->print( log );
    	}
    }

    void printPropertyData( monitoring_logger& log, const std::string& ident ) {
		monitorParameterMap::iterator parameter = parameterData.find( ident);
		if ( parameter != parameterData.cend() ) {
			parameter->second->print( log );
		}
    }

    void printAllData( monitoring_logger& log ) {
    	printCounterData( log );
    	printPropertyData( log );
    }

private:
    explicit monitoring() {
    }

    monitorCounterMap counterData;
    monitorParameterMap parameterData;

};

	#define MONITOR_START_INTERVAL( CTR ) 		morphstore::monitoring::get_instance().startInterval( CTR );
	#define MONITOR_END_INTERVAL( CTR ) 		morphstore::monitoring::get_instance().stopInterval( CTR );
	#define MONITOR_ADD_PROPERTY( key, val )	morphstore::monitoring::get_instance().addProperty( key, val );
	#define MONITOR_PRINT_COUNTERS( ... ) 		morphstore::monitoring::get_instance().printCounterData( __VA_ARGS__ );
	#define MONITOR_PRINT_PROPERTIES( ... )		morphstore::monitoring::get_instance().printPropertyData( __VA_ARGS__ );
	#define MONITOR_PRINT_ALL( ... ) 			morphstore::monitoring::get_instance().printAllData( __VA_ARGS__ );
#else
	#define MONITOR_START_INTERVAL( ... )
	#define MONITOR_END_INTERVAL( ... )
	#define MONITOR_ADD_PROPERTY( ... )
	#define MONITOR_PRINT_COUNTERS( ... )
	#define MONITOR_PRINT_PROPERTIES( ... )
	#define MONITOR_PRINT_ALL( ... )
#endif

}

#endif /* MORPHSTORE_CORE_UTILS_MONITORING_H_ */
