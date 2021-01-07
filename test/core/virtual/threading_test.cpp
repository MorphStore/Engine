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

#include <iostream>

#include <core/virtual/threading.h>
#include <core/virtual/partitioning.h>
#include <vector/vector_extension_structs.h>

template<typename A, typename B, typename C>
struct tmp {
    tmp(A a, B b, C c) : a(a), b(b), c(c){}
    A a;
    B b;
    C c;
};

//struct tmp{
//    tmp(uint64_t a, uint64_t b, uint64_t c): a(a), b(b), c(c) {}
//    uint64_t a;
//    uint64_t b;
//    uint64_t c;
//};

/// This is a very basic select operator with fixed "greater" comparator
/// Do not use out of this testcase
template<typename base_t>
struct op {
    static
    uint64_t * apply(base_t * inDataPtr, uint64_t inDataSize, uint64_t comparatorValue){
        uint64_t * result = new uint64_t;
        for(uint64_t i = 0; i < inDataSize; ++i){
            if(inDataPtr[i] > comparatorValue) {
                *result += 1;
            }
        }
        return result;
    }
};

static bool test_concurrent(){
    bool success = true;
    using namespace vectorlib;
    using namespace virtuallib;
    using namespace std;
    
    using ps = scalar<v64<uint64_t>>;
    using vb = VectorBuilder<cpu_openmp, 4, seq, 1, ps>;
    using base_t = vb::base_t;

    uint64_t elementCount = 10000;
    base_t * testColumn = new base_t[elementCount];
    base_t spin = 0;
    for (uint64_t i = 0; i < elementCount; ++ i) {
        testColumn[i] = spin;
        spin = (spin + 1) % 5;
    }
    
    /// creates one partition per thread
    PartitionSet<base_t> * partitions
      = logical_partitioning<base_t>::apply(testColumn, elementCount, vb::cvalue::value);
//    print_partitions(partitions);
    vector<uint64_t> * comparatorValue = new vector<uint64_t>(vb::cvalue::value, 3);
    
    vector<base_t*> * threadResults
      = concurrent<vb, vb::base_t, op<vb::base_t>, base_t*, uint64_t, uint64_t>
          ::apply(&partitions->dataPtrSet, &partitions->sizeSet, comparatorValue);
    
    if(
      threadResults->size() != 4 or
      *threadResults->at(0) != 500 or
      *threadResults->at(1) != 500 or
      *threadResults->at(2) != 500 or
      *threadResults->at(3) != 500
      ){
      success = false;
    }
    
    return success;
}

int main() {
    using namespace std;
    bool success = true;
    if(!test_concurrent()) {
        cout << "Error. Threading Object concurrent does not work correct." << endl;
        success = false;
    }

    /// .. other tests
    
    if(success)
        cout << "All threading tests are successful." << endl;
    else
        cout << "There are some errors with threading testing." << endl;
        
    return !success;
}
