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

#include <unordered_map>
#include <chrono>
#include <math.h>

#include <core/utils/logger.h>

namespace morphstore {

#ifdef MSV_USE_MONITORING
	#define MONITOR_START_INTERVAL( CTR ) morphstore::monitoring::get_instance().startInterval( #CTR );
	#define MONITOR_END_INTERVAL( CTR ) morphstore::monitoring::get_instance().stopInterval( #CTR );
	#define MONITOR_PRINT_COUNTERS( CTR ) morphstore::monitoring::get_instance().printData( #CTR );
#else
	#define MONITOR_START_INTERVAL( ... )
	#define MONITOR_END_INTERVAL(...)
	#define MONITOR_PRINT_COUNTERS( ... )
#endif

typedef std::chrono::high_resolution_clock::time_point chrono_tp;
typedef std::chrono::high_resolution_clock chronoHighRes;
typedef std::chrono::seconds chronoSec;
typedef std::chrono::milliseconds chronoMilliSec;
typedef std::chrono::microseconds chronoMicroSec;
typedef std::chrono::nanoseconds  chronoNanoSec;

struct monitoring_counter {
	bool started;
	chrono_tp startTp;
	uint32_t maxValues;
	uint32_t lastValue;
	uint64_t* values;

	monitoring_counter() :
		started( false ),
		maxValues( 1024 ),
		lastValue( 0 )
	{
		values = (uint64_t*) malloc( maxValues * sizeof( uint64_t ) );
	}

	void start() {
		startTp = chronoHighRes::now();
		started = true;
	}

	void stop() {
		chrono_tp stopTp = chronoHighRes::now();
		started = false;
		values[ lastValue++ ] = static_cast< uint64_t >( std::chrono::duration_cast< chronoMicroSec >( stopTp - startTp ).count() );
		if ( lastValue == maxValues ) {
			maxValues = std::ceil( maxValues * 1.2 );
			debug( "Reallocating with ", maxValues, " values. Ptr is: ", values  );
			print();
//			uint64_t* newValues = (uint64_t*) realloc( values, maxValues );
			uint64_t* newValues = (uint64_t*) morphstore::stdlib_realloc_ptr( values, maxValues );
			if ( newValues != nullptr ) {
				values = newValues;
			} else {
				throw std::runtime_error("Reallocating of monitoring data array failed");
			}
		}
	}

	void print() {
		for ( size_t i = 0; i < lastValue; ++i ) {
			debug( std::to_string( values[i] ) + " us" );
		}
	}
};

class monitoring {
public:
    static monitoring & get_instance() {
       static thread_local monitoring instance;
       return instance;
    }

    void startInterval( const std::string& ident ) {
    	std::unordered_map< std::string, monitoring_counter* >::iterator counter = data.find( ident);
    	if ( counter != data.cend() ) {
    		if ( ! static_cast< monitoring_counter* >( counter->second )->started ) {
    			static_cast< monitoring_counter* >( counter->second )->start();
    		} else {
    			wtf( "[Monitoring] Starting counter with already open interval: " + ident );
    		}
    	} else {
    		monitoring_counter* ct = new monitoring_counter();
    		data[ ident ] = ct;
    		ct->start();
    	}
    }

    void stopInterval( const std::string& ident ) {
    	std::unordered_map< std::string, monitoring_counter* >::iterator counter = data.find( ident);
    	if ( counter != data.cend() ) {
    		if ( static_cast< monitoring_counter* >( counter->second )->started ) {
    			static_cast< monitoring_counter* >( counter->second )->stop();
    		} else {
    			wtf( "[Monitoring] Stopping counter which never started: " + ident + " -- No timing taken!" );
    		}
    	}
    }

    void printData( const std::string& ident ) {
    	std::unordered_map< std::string, monitoring_counter* >::iterator counter = data.find( ident);
    	if ( counter != data.cend() ) {
    		static_cast< monitoring_counter* >( counter->second )->print();
    	}
    }

private:
    explicit monitoring() {}

    std::unordered_map< std::string, monitoring_counter* > data;

};

}



#endif /* MORPHSTORE_CORE_UTILS_MONITORING_H_ */
