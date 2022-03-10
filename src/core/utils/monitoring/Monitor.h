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


#ifndef MONITOR_H
#define MONITOR_H



#include "MonitorTask.h"
#include <chrono>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <mutex>
#include <core/utils/preprocessor.h>

namespace morphstore {
	namespace monitoring {
		
		enum class MTask {
			__begin__,
			GlobalRuntime,
			Runtime,
			Runtime1,
			Runtime2,
			Runtime3,
			Test1,
			Test2,
			Test3,
			Test4,
			Test5,
			__end__
		};
		
		class Monitor {
		  public:
			Monitor();
			Monitor(const Monitor & orig) = delete;
			~Monitor() = default;
			
			void start(MTask task);
			void resume(MTask task);
			void end(MTask task);
			void reset(MTask task);
			
			
			template<class format>
			uint64_t getDuration(MTask task);
			template<class format>
			std::vector<uint64_t> getAllDurations(MTask task);
			template<class format>
			uint64_t getMean(MTask task);
			template<class format>
			uint64_t getMin(MTask task);
			template<class format>
			uint64_t getMax(MTask task);
			
			std::unordered_map<MTask, MonitorTask> tasks;
			std::unordered_map<MTask, MonitorSeries> series;
			std::map<MTask, std::vector<MonitorTask>> transmittedTasks;
			std::vector<std::vector<MonitorTask>> processedTasks_Best;
			std::vector<std::vector<MonitorTask>> processedTasks_Avrg;
			std::vector<std::vector<MonitorTask>> processedTasks_Wrst;
			
			std::mutex monitorLock;
			
			static Monitor * globalMonitor;
			static Monitor * subRunMonitor;
			static thread_local Monitor * threadMonitor;
			
			static void init();
			static Monitor * getGlobal();
			static Monitor * getThreadLocal();
			
			
			void transmitTo(Monitor *, MTask);
			void transmitToGlobal(MTask);
			
			template<class format>
			uint64_t getMeanTransmitted(MTask);
			template<class format>
			uint64_t getMinTransmitted(MTask);
			template<class format>
			uint64_t getMaxTransmitted(MTask);
			template<class format>
			uint64_t getMedianTransmitted(MTask);
			
			void resetTransmitted(MTask);
		  
		  
		  private:
		};
		
		Monitor* Monitor::globalMonitor = nullptr;
		thread_local Monitor* Monitor::threadMonitor = nullptr;
		
		Monitor::Monitor() {
			for (const auto e : {MTask::__begin__, MTask::__end__}) {
				if (e == MTask::__begin__ or e == MTask::__end__)
					continue;
//				tasks[e] = MonitorTask();
				series[e] = MonitorSeries();
			}
		}
		
		/**
		 * Starts a new measurement for given task.
		 * @param task
		 */
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		void Monitor::start(MTask task) {
			MonitorTask * taskPtr = series[task].newTask();
			taskPtr->active = true;
			taskPtr->begin = clock::now();
		}
		
		/**
		 * Resumes the most recent measurement for given task.
		 * @param task
		 */
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		void Monitor::resume(MTask task) {
			MonitorTask * taskPtr = series[task].getTask();
			if (taskPtr->active)
				return;
			taskPtr->begin = clock::now();
			taskPtr->active = true;
		}
		
		/**
		 * Ends a measurement for given task.
		 * @param task
		 */
		MSV_CXX_ATTRIBUTE_FORCE_INLINE
		void Monitor::end(MTask task) {
			auto end = clock::now();
			MonitorTask * taskPtr = series[task].getTask();
			if(!taskPtr->active)
				return;
			taskPtr->duration = taskPtr->duration + std::chrono::duration_cast<nanoseconds>(end - taskPtr->begin);
			taskPtr->active = false;
		}
        
        
        void Monitor::reset(MTask task) {
            series[task] = MonitorSeries();
        }
        
        /**
         * Returns the measured time from the most recent measurement.
         * @tparam format Wanted format out of std::chrono::[nanoseconds,microseconds,milliseconds,seconds,minutes,hours]
         * @param task
         * @return
         */
		template<class format>
		uint64_t Monitor::getDuration(MTask task) {
			return series[task].get<format>();
		}
		
		/**
		 * Returns the measured times from the whole series.
		 * @tparam format Wanted format out of std::chrono::[nanoseconds,microseconds,milliseconds,seconds,minutes,hours]
		 * @param task
		 * @return
		 */
		template<class format>
		std::vector<uint64_t> Monitor::getAllDurations(MTask task) {
			return series[task].getAll<format>();
		}
		
		template<class format>
		uint64_t Monitor::getMean(MTask task) {
			return series[task].getAvg<format>();
		}
		
		template<class format>
		uint64_t Monitor::getMin(MTask task) {
			return series[task].getMin<format>();
		}
		
		template<class format>
		uint64_t Monitor::getMax(MTask task) {
			return series[task].getMax<format>();
		}
        
        Monitor * Monitor::getGlobal() {
            if(!globalMonitor)
                globalMonitor = new Monitor();
            return globalMonitor;
        }
        
        Monitor * Monitor::getThreadLocal() {
            if(!threadMonitor)
                threadMonitor = new Monitor();
            return threadMonitor;
        }
        
        void Monitor::transmitTo(Monitor * target, MTask type) {
            if(tasks[type].duration.count() > 0) {
                std::lock_guard<std::mutex> lock(target->monitorLock);
                target->transmittedTasks[type].push_back(tasks[type]);
            }
            
        }
        
        void Monitor::transmitToGlobal(MTask type) {
            transmitTo(getGlobal(), type);
        }
        
        template< class format >
        uint64_t Monitor::getMeanTransmitted(MTask type) {
            std::vector<MonitorTask>& tt = transmittedTasks[type];
            std::chrono::nanoseconds avgDuration = std::chrono::nanoseconds(0);
            uint64_t count = 0UL;
            for(auto& t : tt){
                avgDuration += t.duration;
                ++count;
            }
            if(count)
                avgDuration = avgDuration / count;
            return std::chrono::duration_cast<format>(avgDuration);
        }
        
        template< class format >
        uint64_t Monitor::getMinTransmitted(MTask type) {
            std::vector<MonitorTask>& tt = transmittedTasks[type];
            std::chrono::nanoseconds minDuration = std::chrono::nanoseconds(uint64_t(-1));
            for(auto& t : tt){
                if(t.duration < minDuration)
                    minDuration = t.duration;
            }
            return std::chrono::duration_cast<format>(minDuration);
        }
        
        void Monitor::resetTransmitted(MTask type) {
            transmittedTasks[type].clear();
        }
        
    }
}
#endif /* MONITOR_H */

