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
 
#include <logging>
#include <vectorExtensions>
#include <forward>
#include <storage>
#include <vector/general_vector_extension.h>

#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/operators/uncompr/select.h>

using namespace morphstore;
using namespace std;

int main(){
    bool success = true;
    
    const size_t columnSize = 1000;

    column<uncompr_f> * const baseCol1 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            ColumnGenerator::generate_with_distr(
              columnSize,
              std::uniform_int_distribution<uint64_t>(0, 100),
              false,
              8
            )
          )
      );
    
    uint64_t partitionCount = 4;
    uint64_t * data = baseCol1->get_data();
    
    std::cout << "Pointer to data: " << (uint64_t) data << std::endl;
    PartitionedColumn<uncompr_f, uint64_t, LogicalPartitioner<column<uncompr_f>, uint64_t>> pc(baseCol1, partitionCount);
    
    /// check partition 0
    if( pc[0]->get_count_values() != 256 or ( ((uint64_t*)pc[0]->get_data()) != data ) ) {
        success = false;
        error("In partition 0");
    }
    /// check partition 1
    if( pc[1]->get_count_values() != 256 or ( ((uint64_t*)pc[1]->get_data()) != (data + 256) ) ) {
        success = false;
        error("In partition 1");
    }
    /// check partition 2
    if( pc[2]->get_count_values() != 256 or ( ((uint64_t*)pc[2]->get_data()) != (data + 256 + 256) ) ) {
        success = false;
        error("In partition 2");
    }
    /// check partition 3
    if( pc[3]->get_count_values() != 232 or ( ((uint64_t*)pc[3]->get_data()) != (data + 256 + 256 + 256) ) ) {
        success = false;
        error("In partition 3");
    }
    
    /// sanity check
    cout << "Partition 0: " << pc[0]->get_count_values() << " values  / " << (uint64_t) (uint64_t*) pc[0]->get_data() << endl;
    cout << "Partition 1: " << pc[1]->get_count_values() << " values  / " << (uint64_t) (uint64_t*) pc[1]->get_data() << " diff: " << (uint64_t)( ((uint64_t*)pc[1]->get_data() - (uint64_t*)pc[0]->get_data()) * sizeof(uint64_t) ) << "B" << endl;
    cout << "Partition 2: " << pc[2]->get_count_values() << " values  / " << (uint64_t) (uint64_t*) pc[2]->get_data() << " diff: " << (uint64_t)( ((uint64_t*)pc[2]->get_data() - (uint64_t*)pc[1]->get_data()) * sizeof(uint64_t) ) << "B" << endl;
    cout << "Partition 3: " << pc[3]->get_count_values() << " values  / " << (uint64_t) (uint64_t*) pc[3]->get_data() << " diff: " << (uint64_t)( ((uint64_t*)pc[3]->get_data() - (uint64_t*)pc[2]->get_data()) * sizeof(uint64_t) ) << "B" << endl;
    
    
    return !success;
}
