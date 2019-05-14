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

#include <iostream>
#include <algorithm>
#include <tuple>
#include <atomic>
#include <functional>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <math.h>
#include <inttypes.h>

#include <core/utils/logger.h>
#include <core/utils/preprocessor.h>

namespace morphstore {

#ifdef MSV_USE_MONITORING
	typedef std::chrono::high_resolution_clock::time_point chrono_tp;
	typedef std::chrono::high_resolution_clock chronoHighRes;
	typedef std::chrono::seconds chronoSec;
	typedef std::chrono::milliseconds chronoMilliSec;
	typedef std::chrono::microseconds chronoMicroSec;
	typedef std::chrono::nanoseconds  chronoNanoSec;


	namespace print {
		using namespace std;

		template<int index, typename... Ts>
		struct printTupleRecursive {
			std::string operator() (char delim, const tuple<Ts...>& t) {
				std::stringstream ss;
				ss << get<index>(t) << delim;
				ss << printTupleRecursive<index - 1, Ts...>{}(delim, t);
				return ss.str();
			}
		};

		template<typename... Ts>
		struct printTupleRecursive<0, Ts...> {
			std::string operator() (char delim, const tuple<Ts...>& t) {
				std::stringstream ss;
				ss << get<0>(t);
				return ss.str();
			}
		};

		template<typename... Ts>
		std::string printTuple(char delim, const tuple<Ts...>& t) {
			const auto size = tuple_size<tuple<Ts...>>::value;
			return printTupleRecursive<size - 1, Ts...>{}(delim, t);
		}
	}

	namespace compare {
		template<int index, typename... Ts>
		struct compareTuplesRecursive {
			bool operator() (std::tuple<Ts...>& t, std::tuple<Ts...>& u) {
				return (std::get<index>(t) == std::get<index>(u)) && compareTuplesRecursive<index - 1, Ts...>{}(t, u);
			}
		};

		template<typename... Ts>
		struct compareTuplesRecursive<0, Ts...> {
			bool operator() (std::tuple<Ts...>& t, std::tuple<Ts...>& u) {
				return std::get<0>(t) == std::get<0>(u);
			}
		};

		template<typename... Ts>
		bool compareTuples(std::tuple<Ts...>& t, std::tuple<Ts...>& u) {
			const auto size = std::tuple_size<std::tuple<Ts...>>::value;
			return compareTuplesRecursive<size - 1, Ts...>{}(t, u);
		}
	}

	class monitoring_info {
	public:
		const std::string name;
		size_t id;

		monitoring_info(const std::string name, size_t id) :
			name(name),
			id(id)
		{
		}

		virtual ~monitoring_info() {
		}

		std::string getName() const {
			return name;
		}

		virtual std::string getAsString(const size_t idx) const = 0;
		//virtual void print(monitoring_logger& log) = 0;
	};

	template< typename T >
	class monitoring_parameter : public monitoring_info {
	public:
		uint32_t maxValues;
		uint32_t lastValue;
		T* values;

		monitoring_parameter(const std::string name, size_t id, T value) :
			monitoring_info(name, id),
			maxValues(1024),
			lastValue(0)
		{
			values = (T*) malloc( maxValues * sizeof(T) );
			addValue(value);
		}

		T getValue( const uint32_t idx ) const {
			return values[ idx ];
		}

		void addValue(const T val) {
			values[lastValue++] = val;
			if (lastValue == maxValues) {
				maxValues = std::ceil(maxValues * 1.2);
				T* newValues = (T*)realloc(values, maxValues);
				if (newValues != nullptr) {
					values = newValues;
				} else {
					throw std::runtime_error("Reallocating of monitoring data array failed.");
				}
			}
		}

		std::string getAsString(const size_t idx) const override {
			if (idx >= lastValue) {
				throw std::runtime_error("[MONITORING ERROR] Trying to get a parameter with out-of-bounds index. Did you add an uneven amount of parameters/intervals?");
			}
			return std::to_string( values[idx] );
		}

		/*void print(monitoring_logger& log) {
			log.write("[Monitoring] ");
			log.write(name.c_str());
			log.write(" (id ");
			log.write(id);
			log.write("): ");
			log.write(value);
			log.write("\n");
		}*/
	};

	class monitoring_counter : public monitoring_info {
	public:
		bool started;
		chrono_tp startTp;
		uint32_t maxValues;
		uint32_t lastValue;
		uint64_t* values;

		explicit monitoring_counter(const std::string name, size_t id) :
			monitoring_info(name, id),
			started(false),
			maxValues(1024),
			lastValue(0)
		{
			values = (uint64_t*)malloc(maxValues * sizeof(uint64_t));
		}

		void start() {
			started = true;
			startTp = chronoHighRes::now();
		}

		void stop(chrono_tp stopTp) {
			started = false;
			values[lastValue++] = static_cast<uint64_t>(std::chrono::duration_cast<chronoMicroSec>(stopTp - startTp).count());
			if (lastValue == maxValues) {
				maxValues = std::ceil(maxValues * 1.2);
				/*debug("Reallocating with ", maxValues, " values. Ptr is: ", values);*/
				uint64_t* newValues = (uint64_t*)realloc(values, maxValues);
				if (newValues != nullptr) {
					values = newValues;
				}
				else {
					throw std::runtime_error("Reallocating of monitoring data array failed.");
				}
			}
		}

		std::string getAsString(const size_t idx) const override {
			if (idx >= lastValue) {
				throw std::runtime_error("[MONITORING ERROR] Trying to get an interval with out-of-bounds index. Did you add an uneven amount of parameters/intervals?");
			}
			return std::to_string( values[idx] );
		}
	};

	struct CompareMonitorInfo {
		bool operator() (const monitoring_info* left, const monitoring_info* right)
		{
			return left->id < right->id;
		}
	} CompareMonitorInfo;


	typedef std::map< std::string, monitoring_counter* > monitorIntervalMap;
	typedef std::map< std::string, monitoring_parameter<bool>* > monitorBoolParameterMap;
	typedef std::map< std::string, monitoring_parameter<int64_t>* > monitorIntegerParameterMap;
	typedef std::map< std::string, monitoring_parameter<double>* > monitorDoubleParameterMap;

	class SuperMon {
	public:
		explicit SuperMon(size_t id) :
			id(id)
		{
		}

		virtual ~SuperMon() {};

		virtual std::string getTupleAsString(char delim) const = 0;
		virtual void addHeadKeys(const std::vector< std::string >& hKeys) = 0;

		void startInterval(const std::string& ident) {
			monitorIntervalMap::iterator counter = intervalData.find(ident);
			if (counter != intervalData.cend()) {
				if (!counter->second->started) {
					counter->second->start();
				}
				else {
					//wtf("[Monitoring] Starting counter with already open interval: " + ident);
					std::cout << "[Monitoring] Error - Starting counter with already open interval: " + ident << std::endl;
				}
			}
			else {
				monitoring_counter* ct = new monitoring_counter(ident, rollingId++);
				intervalData.insert({ ident, ct });
				ct->start();
			}
		}

		void stopInterval(const std::string& ident) {
			chrono_tp stopTp = chronoHighRes::now();
			monitorIntervalMap::iterator counter = intervalData.find(ident);
			if (counter != intervalData.cend()) {
				if (counter->second->started) {
					counter->second->stop(stopTp);
				}
				else {
					//wtf("[Monitoring] Stopping counter which never started: " + ident + " -- No timing taken!");
					std::cout << "[Monitoring] Error - Stopping counter which never started: " + ident + " -- No timing taken!" << std::endl;
				}
			}
		}

		void addBoolProperty(const std::string& ident, const bool val) {
			monitorBoolParameterMap::iterator counter = boolParams.find(ident);
			if (counter != boolParams.cend()) {
				counter->second->addValue(val);
			}
			else {
				boolParams.insert({ ident, new monitoring_parameter<bool>(ident, rollingId++, val) });
			}
		}

		void addIntegerProperty(const std::string& ident, const int64_t val) {
			monitorIntegerParameterMap::iterator counter = integerParams.find(ident);
			if (counter != integerParams.cend()) {
				counter->second->addValue(val);
			}
			else {
				integerParams.insert({ ident, new monitoring_parameter<int64_t>(ident, rollingId++, val) });
			}
		}

		//void incrementIntegerProperty(const std::string& ident, const int64_t val) {
		//	monitorIntegerParameterMap::iterator counter = integerParams.find(ident);
		//	if (counter != integerParams.end()) {
		//		std::cout << "[Monitoring] Updated integer parameter " << ident << " old value: " << counter->second->getValue();
		//		counter->second->value = counter->second->value + val;
		//		std::cout << " new value: " << counter->second->value << std::endl;
		//	}
		//	else {
		//		std::cout << "[ERROR] No matching parameter, done nothing." << std::endl;
		//	}
		//}

		void addDoubleProperty(const std::string& ident, const double val) {
			monitorDoubleParameterMap::iterator counter = doubleParams.find(ident);
			if (counter != doubleParams.cend()) {
				counter->second->addValue(val);
			}
			else {
				doubleParams.insert({ ident, new monitoring_parameter<double>(ident, rollingId++, val) });
			}
		}

		//void incrementDoubleProperty(const std::string& ident, const double val) {
		//	monitorDoubleParameterMap::iterator counter = doubleParams.find(ident);
		//	if (counter != doubleParams.end()) {
		//		std::cout << "[Monitoring] Updated double parameter " << ident << " old value: " << counter->second->getValue();
		//		counter->second->value = counter->second->value + val;
		//		std::cout << " new value: " << counter->second->value << std::endl;
		//	}
		//	else {
		//		std::cout << "[ERROR] No matching parameter, done nothing." << std::endl;
		//	}
		//}

		template< class T >
		void insertSorted(T& fromMap, std::vector< monitoring_info* >& toVec) const {
			for (auto info : fromMap) {
				toVec.insert(std::upper_bound(toVec.begin(), toVec.end(), info.second, CompareMonitorInfo), info.second);
			}
			//std::for_each(std::begin(fromMap), std::end(fromMap), [&](const auto& x) {toVec.insert(std::upper_bound(toVec.begin(), toVec.end(), x.second, CompareMonitorInfo), x.second); });
		}

		std::vector< monitoring_info* > createSortedCounterList() const {
			std::vector< monitoring_info* > sortedInfo;
			sortedInfo.reserve(intervalData.size() + boolParams.size() + integerParams.size() + doubleParams.size());
			insertSorted(intervalData, sortedInfo);
			insertSorted(boolParams, sortedInfo);
			insertSorted(integerParams, sortedInfo);
			insertSorted(doubleParams, sortedInfo);
			return sortedInfo;
		}

		uint32_t getMaxLines() {
			uint32_t maxLines = 0;
			for (auto ct : intervalData) {
				maxLines = std::max(maxLines, ct.second->lastValue);
			}
			for (auto ct : boolParams) {
				maxLines = std::max(maxLines, ct.second->lastValue);
			}
			for (auto ct : integerParams) {
				maxLines = std::max(maxLines, ct.second->lastValue);
			}
			for (auto ct : doubleParams) {
				maxLines = std::max(maxLines, ct.second->lastValue);
			}

			return maxLines;
		}

		const monitorIntervalMap& getIntervals() const {
			return intervalData;
		}

		const monitorBoolParameterMap& getBoolParams() const {
			return boolParams;
		}

		const monitorDoubleParameterMap& getDoubleParams() const {
			return doubleParams;
		}

		const monitorIntegerParameterMap& getIntegerParams() const {
			return integerParams;
		}

		virtual std::string getAllheads(char delim) const = 0;

		std::string printAllData(/*monitoring_logger& log, */char delim, const size_t idx ) const {
			std::vector< monitoring_info* > sortedInfo = createSortedCounterList();

			std::stringstream ss;
			ss << getTupleAsString(delim);
			for (size_t i = 0; i < sortedInfo.size(); ++i) {
				ss << sortedInfo[i]->getAsString(idx);
				if (i < sortedInfo.size() - 1) {
					ss << delim;
				}
			}
			return ss.str();
		}

		size_t id;

	protected:
		std::atomic< size_t > rollingId = { 0 };

		monitorIntervalMap intervalData;
		monitorBoolParameterMap boolParams;
		monitorIntegerParameterMap integerParams;
		monitorDoubleParameterMap doubleParams;
	};

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
	virtual void log(std::vector< SuperMon* > monitors) = 0;
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

	virtual void log(MSV_CXX_ATTRIBUTE_PPUNUSED std::vector< SuperMon* > monitors) {}

	static monitoring_shell_logger & get_instance(void) {
		static monitoring_shell_logger instance;
		return instance;
	}
};

class monitoring_csv_logger : public monitoring_logger {
private:
	monitoring_csv_logger() {
	}

public:
	void write(const char* text) override {
		printf("%s", text);
	}

	void write(uint64_t val) override {
		printf("%" PRIu64, val);
	}

	void write(double val) override {
		printf("%f", val);
	}

	virtual void log(std::vector< SuperMon* > monitors) {
		std::cout << monitors[0]->getAllheads('\t') << std::endl;
		size_t maxLines = monitors[0]->getMaxLines();
		for (size_t idx = 0; idx < maxLines; ++idx) {
			for (auto m : monitors) {
				std::cout << m->printAllData('\t', idx);
				std::cout << std::endl;
			}
		}
	}

	static monitoring_csv_logger  & get_instance(void) {
		static monitoring_csv_logger  instance;
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

	virtual void log(MSV_CXX_ATTRIBUTE_PPUNUSED std::vector< SuperMon* > monitors) {}

	static monitoring_file_logger & get_instance(void) {
		static monitoring_file_logger instance;
		return instance;
	}
};

//struct SimpleJson {
//	std::string json;
//	SimpleJson() {
//	}
//
//	void startEntity() {
//		json += "{";
//	}
//	void endEntity() {
//		json += "}";
//	}
//	void startArray() {
//		json += "[";
//	}
//	void endArray() {
//		json += "]";
//	}
//
//	template< typename T >
//	void addValue( const T& value) {
//		if (!firstValue) {
//			json += ",";
//		}
//		else {
//			firstValue = false;
//		}
//		json += "\"" + ident + ":" + std::to_string(value);
//	}
//
//	template< typename T >
//	void addKeyValue(const std::string& ident, const T& value) {
//		if (!firstValue) {
//			json += ",";
//		}
//		else {
//			firstValue = true;
//		}
//		json += "\"" + ident + ":" + std::to_string(value);
//	}
//
//	bool firstEntity = true;
//	bool firstArray =  true;
//	bool firstValue =  true;
//};

class monitoring_json_logger : public monitoring_logger {
private:
	std::string* logFileName;

	monitoring_json_logger() {
		auto now = std::chrono::system_clock::now();
		auto in_time_t = std::chrono::system_clock::to_time_t(now);

		std::stringstream ss;
		ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X_monitoringLog");
		logFileName = new std::string(ss.str().c_str());
		std::cout << "JSonLogFilename: " << *logFileName << std::endl;
	}

public:
	static monitoring_json_logger & get_instance(void) {
		static monitoring_json_logger instance;
		return instance;
	}

	void log( const std::vector< SuperMon* >& monitors) {
		std::cout << "Logging for json" << std::endl;
		for (auto mon : monitors) {
			std::cout << " Mon " << mon->id << std::endl;
		}
	}

	void write(MSV_CXX_ATTRIBUTE_PPUNUSED const char* text) override {};
	void write(MSV_CXX_ATTRIBUTE_PPUNUSED uint64_t val) override {};
	void write(MSV_CXX_ATTRIBUTE_PPUNUSED double val) override {};

	virtual void log(MSV_CXX_ATTRIBUTE_PPUNUSED std::vector< SuperMon* > monitors) {
		/*std::stringstream json = "[";
		
		for (size_t idx = 0; idx < monitors.size(); ++idx ) {
			SuperMon* mon = monitors[idx];
			
			json << "{";

			for (const auto bParam : mon->getBoolParams()) {
				json << "\"" << bParam.first << "\":[" << bParam.second->getAsString() << "]";
			}
		}

		json << "}";
		json << "]";*/
	}
};

auto monitorShellLog = morphstore::monitoring_shell_logger::get_instance();
auto monitorFileLog = morphstore::monitoring_file_logger::get_instance();
auto monitorJSonLog = morphstore::monitoring_json_logger::get_instance();
auto monitorCsvLog = morphstore::monitoring_csv_logger::get_instance();


template <typename... T>
class Monitor : public SuperMon {
public:

	template <typename... Ts>
	Monitor(size_t id, Ts... args) :
		SuperMon(id),
		key(std::forward<Ts>(args)...)
	{ }

	~Monitor() {
	}

	std::string getAllheads(char delim) const override {
		std::stringstream ss;
		for (size_t i = 0; i < keyHeads.size(); ++i) {
			ss << keyHeads[i] << delim;
		}
		auto sortedInfo = createSortedCounterList();
		for (size_t i = 0; i < sortedInfo.size(); ++i) {
			ss << sortedInfo[i]->getName();
			if (i < sortedInfo.size() - 1) {
				ss << delim;
			}
		}
		return ss.str();
	}

	std::string getTupleAsString(char delim) const override {
		// fold-expressions only in c++1z / c++17
		return callDoPrint(delim, key, std::index_sequence_for<T...>());
		//return print::printTuple(delim, key);
	}

	void addHeadKeys(const std::vector< std::string >& hKeys) override {
		for (const std::string& s : hKeys) {
			keyHeads.push_back(s);
		}
	}

	// fold-expressions only in c++1z / c++17
	std::string doPrint(char delim, T... values) const {
		//((std::cout << (values) << " "),...);
		std::stringstream ss;
		using isoHelper = int[];
		(void)isoHelper {
			0, (void(ss << std::forward< T >(values) << delim), 0) ...
		};
		return ss.str();
	}

	// fold-expressions only in c++1z / c++17
	template<std::size_t... Is>
	std::string callDoPrint(char delim, const std::tuple<T...>& tuple, std::index_sequence<Is...>) const {
		return doPrint(delim, std::get<Is>(tuple)...);
	}

	std::tuple< T... > key;
	std::vector< std::string > keyHeads;
};

class Monitoring {
public:
	static Monitoring & get_instance() {
		static thread_local Monitoring instance;
		return instance;
	}

	template <typename... Args>
	Monitor<Args...>* make_mon(Args... args)
	{
		return new Monitor<Args...>(rollingMonitorId++, std::forward<Args>(args)...);
	}

	void clearAll() {
		for (auto m : monVec) {
			delete m;
		}
		monVec.clear();
	}

	template<typename... Ts>
	SuperMon* findMonitor(Ts... args) {
		std::tuple< Ts... > lookup(args...);
		for (size_t i = 0; i < monVec.size(); ++i) {
			std::tuple< Ts... > castedKey = static_cast<std::tuple< Ts... >>(static_cast<Monitor< Ts... >*>(monVec[i])->key);
			if (compare::compareTuples(lookup, castedKey)) {
				return monVec[i];
			}
		}
		return nullptr;
	}

	void addKeyHeads(std::vector< std::string >& headVec, std::string& head) {
		headVec.push_back(head);
	}

	template< typename... HeadVals >
	void addKeyHeads(std::vector< std::string >& headVec, std::string& head, HeadVals&... heads) {
		addKeyHeads(headVec, head);
		addKeyHeads(headVec, heads...);
	}

	/*template< typename... headVals >
	void create(SuperMon* mon, const std::string& head, headVals... heads) {
		std::vector< std::string > headKeys;
		addKeyHeads(headKeys, head, heads...);
		for (size_t i = 0; i < headkeys.size(); ++i) {
			std::cout << "######### HEAD KEY " << headkeys[i] << std:.endl;
		}
	}*/

	void create(SuperMon* mon, const std::vector< std::string >& headKeys) {
		mon->addHeadKeys(headKeys);
	}

	template<typename... Ts>
	SuperMon* createMonitor(Ts ... args) {
		auto mon = findMonitor(args...);
		if (mon == nullptr) {
			mon = make_mon(args...);
			//std::cout << "Created monitor (" << mon << ") with key { " << mon->printTuple('-') << " | id " << mon->id << " } and pushed it to the vector" << std::endl;
			monVec.push_back(mon);
		}
		else {
			throw std::runtime_error("[MONITORING ERROR] Trying to add a monitor with already existing key.");
		}
		return mon;
	}

	template<typename... Ts>
	void startIntervalFor(std::string ident, Ts... args) {
		auto mon = findMonitor(args...);
		if (mon) {
			mon->startInterval(ident);
		}
		else {
			throw std::runtime_error("[MONITORING ERROR] Trying to start an interval for a non-existent monitor.");
		}
	}

	template<typename... Ts>
	void endIntervalFor(std::string ident, Ts... args) {
		auto mon = findMonitor(args...);
		if (mon) {
			mon->stopInterval(ident);
		}
		else {
			throw std::runtime_error("[MONITORING ERROR] Trying to stop an interval for a non-existent monitor.");
		}
	}

	template<typename... Ts>
	void addBoolFor(std::string ident, bool val, Ts... args) {
		SuperMon* mon = findMonitor(args...);
		if (mon) {
			mon->addBoolProperty(ident, val);
		}
		else {
			throw std::runtime_error("[MONITORING ERROR] Trying to add an bool parameter for a non-existent monitor.");
		}
	}

	template<typename... Ts>
	void addIntFor(std::string ident, int64_t val, Ts... args) {
		SuperMon* mon = findMonitor(args...);
		if (mon) {
			mon->addIntegerProperty(ident, val);
		}
		else {
			throw std::runtime_error("[MONITORING ERROR] Trying to add an int parameter for a non-existent monitor.");
		}
	}

	template<typename... Ts>
	void addDoubleFor(std::string ident, double val, Ts... args) {
		SuperMon* mon = findMonitor(args...);
		if (mon) {
			mon->addDoubleProperty(ident, val);
		}
		else {
			throw std::runtime_error("[MONITORING ERROR] Trying to add a double parameter for a non-existent monitor.");
		}
	}

	//template<typename... Ts>
	//void incrementIntFor(std::string ident, int64_t val, Ts... args) {
	//	SuperMon* mon = findMonitor(args...);
	//	if (mon) {
	//		mon->incrementIntegerProperty(ident, val);
	//	}
	//	else {
	//		throw std::runtime_error("[MONITORING ERROR] Trying to increment an int parameter for a non-existent monitor.");
	//	}
	//}

	//template<typename... Ts>
	//void incrementDoubleFor(std::string ident, double val, Ts... args) {
	//	SuperMon* mon = findMonitor(args...);
	//	if (mon) {
	//		mon->incrementDoubleProperty(ident, val);
	//	}
	//	else {
	//		throw std::runtime_error("[MONITORING ERROR] Trying to increment a double parameter for a non-existent monitor.");
	//	}
	//}

	template< typename... Ts >
	void printMonitor(Ts... args) {
		SuperMon* mon = findMonitor(args...);
	}

	void printAll( monitoring_logger& logger ) {
		/*std::cout << "Printing all monitor keys" << std::endl;
		std::cout << monVec[0]->getAllheads('\t') << std::endl;
		for (auto m : monVec) {
			std::cout << m->printAllData('\t');
			std::cout << std::endl;
		}*/
		logger.log(monVec);
	}

	std::atomic< size_t > rollingMonitorId = { 0 };
	std::vector< SuperMon* > monVec;
};

	#define MONITORING_CREATE_MONITOR( ... ) 					Monitoring::get_instance().create( __VA_ARGS__ );
	#define MONITORING_MAKE_MONITOR( ... ) 						Monitoring::get_instance().createMonitor( __VA_ARGS__ )
	#define MONITORING_KEY_IDENTS( ... ) 						std::vector< std::string > { __VA_ARGS__ }
	#define MONITORING_START_INTERVAL_FOR( ident, ... ) 		Monitoring::get_instance().startIntervalFor( ident, __VA_ARGS__ )
	#define MONITORING_END_INTERVAL_FOR( ident, ... ) 			Monitoring::get_instance().endIntervalFor( ident, __VA_ARGS__ )
	#define MONITORING_ADD_BOOL_FOR( ident, val, ... ) 			Monitoring::get_instance().addBoolFor( ident, val, __VA_ARGS__ )
	#define MONITORING_ADD_INT_FOR( ident, val, ... ) 			Monitoring::get_instance().addIntFor( ident, val, __VA_ARGS__ )
	#define MONITORING_ADD_DOUBLE_FOR( ident, val, ... ) 		Monitoring::get_instance().addDoubleFor( ident, val, __VA_ARGS__ )
	/*#define MONITORING_INCREMENT_INT_BY( ident, val, ... )		Monitoring::get_instance().incrementIntFor( ident, val, __VA_ARGS__ )
	#define MONITORING_INCREMENT_DOUBLE_BY( ident, val, ... )	Monitoring::get_instance().incrementDoubleFor( ident, val, __VA_ARGS__ )*/
	#define MONITORING_PRINT_MONITOR( ... )						Monitoring::get_instance().printMonitor( __VA_ARGS__ );
	#define MONITORING_PRINT_MONITORS(logger)					Monitoring::get_instance().printAll( logger )
	#define MONITORING_CLEAR_ALL()								Monitoring::get_instance().clearAll()
	#define MONITORING_INIT_CRITICAL_TIMING()					std::vector< std::pair< std::string, uint64_t > > criticalTimings
	#define MONITORING_START_CRITICAL_TIMING( timer )			auto startTp_##timer =std::chrono::high_resolution_clock::now()
	#define MONITORING_END_CRITICAL_TIMING( timer )				\
																auto endTp_##timer =std::chrono::high_resolution_clock::now(); \
criticalTimings.emplace_back( \
std::make_pair(\
"timer", \
std::chrono::duration_cast<std::chrono::nanoseconds>(endTp_##timer - startTp_##timer).count() \
																		)  \
																	);
	#define MONITORING_FINALIZE_CRITICAL_TIMINGS()				for ( auto p : criticalTimings ) { std::cout << p.first << "\t" << p.second << std::endl; }
#else
	#define MONITORING_CREATE_MONITOR( ... )
	#define MONITORING_MAKE_MONITOR( ... )	
	#define MONITORING_START_INTERVAL_FOR( ... )
	#define MONITORING_END_INTERVAL_FOR( ... )
	#define MONITORING_ADD_BOOL_FOR( ... )
	#define MONITORING_ADD_INT_FOR( ... ) 
	#define MONITORING_ADD_DOUBLE_FOR( ... )
	#define MONITORING_INCREMENT_INT_BY( ... )
	#define MONITORING_INCREMENT_DOUBLE_BY( ... )
	#define MONITORING_PRINT_MONITOR( ... )	
	#define MONITORING_PRINT_MONITORS( ... )
	#define MONITORING_CLEAR_ALL()
	#define MONITORING_INIT_CRITICAL_TIMING()
	#define MONITORING_START_CRITICAL_TIMING( ... )
	#define MONITORING_END_CRITICAL_TIMING( ... )
	#define MONITORING_FINALIZE_CRITICAL_TIMINGS()
#endif

}

#endif /* MORPHSTORE_CORE_UTILS_MONITORING_H_ */
