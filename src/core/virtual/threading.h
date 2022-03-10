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


#ifndef MORPHSTORE_THREADING_H
#define MORPHSTORE_THREADING_Hs

#include <type_traits>
//#include <thread>
//#include <vector>
#include <stdlibs>


namespace virtuallib {
    /// pthread implementation
    /// @todo @eric add type
    template <class TVectorBuilder, class TReturn, class TOperator, typename ... TInputTypes>
    struct concurrent {
        using base_t = typename TVectorBuilder::base_t;
        
        static
        void lambda(TReturn ** output, TInputTypes&& ... input){
            /// output is an array of pointers, one entry per thread
            /// each thread writes the resulting output pointer to its assigned cell
            *output = TOperator::apply(std::forward<TInputTypes>(input) ... );
        }
        
        static
        std::vector<TReturn*>* apply(std::vector<TInputTypes>* ... vecs){
            const uint16_t threadCnt = TVectorBuilder::cvalue::value;
			/// thread container
			std::thread * threads[threadCnt];
			
			auto results = new std::vector<TReturn*>(threadCnt);
			/// init threads
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
			    TReturn ** output = &(*results)[threadIdx];
//			    threads[threadIdx] = new std::thread(lambda, output, std::forward<TArgs>(args) ... );
			    threads[threadIdx] = new std::thread(lambda, output, vecs->at(threadIdx) ... );
//			    threads[threadIdx] = new std::thread(lambda, output, 3 );
//			    threads[threadIdx] = new std::thread(lambda, output, args ... );
			}
			
			/// wait for threads to finish
			for(uint16_t threadIdx = 0; threadIdx < threadCnt; ++threadIdx){
				threads[threadIdx]->join();
				delete threads[threadIdx];
			}
   
			return results;
        }
    };
}

#endif //MORPHSTORE_THREADING_H
