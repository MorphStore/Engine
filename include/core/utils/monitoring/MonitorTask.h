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


#ifndef QUEUEBENCHMARK_MONITORTASK_H
#define QUEUEBENCHMARK_MONITORTASK_H

#include <chrono>
#include <vector>
#include <cstdlib>
#include <iostream>

namespace morphstore {
	namespace monitoring {
		
		using clock        = std::chrono::high_resolution_clock;
		using nanoseconds  = std::chrono::nanoseconds;
		using microseconds = std::chrono::microseconds;
		using milliseconds = std::chrono::milliseconds;
		using seconds      = std::chrono::seconds;
		using minutes      = std::chrono::minutes;
		using hours        = std::chrono::hours;
		
		/**
		 * Type trait check if class is a std::chrono time format
		 * @tparam T Class to check
		 */
		template<class T>
	    struct is_chrono_format : public std::false_type {};
		template<>
	    struct is_chrono_format<nanoseconds> : public std::true_type {};
		template<>
	    struct is_chrono_format<microseconds> : public std::true_type {};
		template<>
	    struct is_chrono_format<milliseconds> : public std::true_type {};
		template<>
	    struct is_chrono_format<seconds> : public std::true_type {};
		template<>
	    struct is_chrono_format<minutes> : public std::true_type {};
		template<>
	    struct is_chrono_format<hours> : public std::true_type {};
		
		
		/**
		 *
		 */
		struct MonitorTask {
			std::chrono::nanoseconds duration = std::chrono::nanoseconds(0);
			std::chrono::system_clock::time_point begin;
			bool active = false;
			
			void operator+=(MonitorTask & other) {
				duration += other.duration;
			}
			
			void operator/=(unsigned divisor) {
				duration = duration / divisor;
			}
		};
		
		
		/**
		 *
		 */
		class MonitorSeries {
			std::vector<MonitorTask> tasks;
			
			MonitorTask * currentTask = nullptr;
			
		  public:
			MonitorTask * newTask() {
				tasks.emplace_back();
				currentTask = &(*(--tasks.end()));
				return currentTask;
			}
			
			MonitorTask * getTask() {
				if(!currentTask)
					return newTask();
				return currentTask;
			}
			
			template< typename format/*, typename std::enable_if<is_chrono_format<format>::value, format>::type = 0 */>
			uint64_t get() {
				static_assert(is_chrono_format<format>::value, "Specified format is not a chrono time format!");
				if(!currentTask)
					return std::chrono::duration_cast<format>(nanoseconds(0)).count();
				return std::chrono::duration_cast<format>(currentTask->duration).count();
			}
			
			template<class format>
			uint64_t getMin() {
				static_assert(is_chrono_format<format>::value, "Specified format is not a chrono time format!");
				nanoseconds min(0x0fffffffffffffff);
				for(auto & task : tasks) {
					min = std::min(min, task.duration);
//					if(task.duration < min)
//						min = task.duration;
				}
				return std::chrono::duration_cast<format>(min).count();
			}
			
			template<class format>
			uint64_t getMax() {
				static_assert(is_chrono_format<format>::value, "Specified format is not a chrono time format!");
				nanoseconds max(0);
				for(auto & task : tasks) {
					max = std::max(max, task.duration);
				}
				return std::chrono::duration_cast<format>(max).count();
			}
			
			template<class format>
			uint64_t getAvg() {
				static_assert(is_chrono_format<format>::value, "Specified format is not a chrono time format!");
				nanoseconds avg(0);
				
				for(auto & task : tasks) {
					avg += task.duration;
				}
				if(!tasks.empty())
					avg /= tasks.size();
				return std::chrono::duration_cast<format>(avg).count();
			}
			
			template<class format>
			std::vector<uint64_t> getAll() {
				static_assert(is_chrono_format<format>::value, "Specified format is not a chrono time format!");
				std::vector<uint64_t> result;
				for(auto t : tasks){
					result.push_back(std::chrono::duration_cast<format>(t.duration).count());
				}
				return std::move(result);
				
//				uint64_t size = tasks.size();
//				uint64_t * result = (uint64_t*) malloc(size * sizeof(uint64_t));
//
//				for (uint64_t i = 0; i < size; ++i) {
//					result[i] = std::chrono::duration_cast<format>(tasks[i].duration).count();
//				}
//				return std::make_pair(size,result);
			}
			
			template<class format>
			std::vector<format> getAllRaw() {
				static_assert(is_chrono_format<format>::value, "Specified format is not a chrono time format!");
				std::vector<format> result;
				for(auto t : tasks) {
					result.push_back(std::chrono::duration_cast<format>(t.duration));
				}
				return std::move(result);
				
//				uint64_t size = tasks.size();
//				format * result = (format*) malloc(size * sizeof(format));
//				for (uint64_t i = 0; i < size; ++i) {
//					result[i] = std::chrono::duration_cast<format>(tasks[i].duration);
//				}
//				return std::make_pair(size,result);
			}
		};
	}
}
#endif //QUEUEBENCHMARK_MONITORTASK_H
