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
#include <forward>

#include <core/utils/logger.h>
/// primitives
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <storage>
#include <virtual>

#include <operators>

#include <core/operators/uncompr/select.h>
#include <core/storage/Partitioner.h>

#include <core/virtual/MetaOperator.h>

#include <printing>



struct op : Executable {
    static
    uint64_t apply(column<uncompr_f> * const, uint64_t * ){
        std::cout << "called op::apply()" << std::endl;
        return 0;
    }
};
struct op2 : Executable {
    static
    uint64_t apply(column<uncompr_f> * const, uint64_t ){
        std::cout << "called op2::apply()" << std::endl;
        return 0;
    }
};


int main() {
    using namespace morphstore;
    using namespace vectorlib;
    using namespace virtuallib;
    
    /// 2 GB of data
//    const size_t countValues = 1024 * 1024 / sizeof(uint64_t) * 2 * 1024;
//    const size_t countValues = sizeof(uint64_t) * 2 * 1024;
    const size_t countValues = sizeof(uint64_t) * 2 * 10;

    const column<uncompr_f> * const baseCol1 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            ColumnGenerator::generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                8
            )
          )
      );
    const column<uncompr_f> * const baseCol2 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            ColumnGenerator::generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                9
            )
          )
      );

    const uint64_t threadCnt = 2;

    using ve = scalar<v64<uint64_t>>;
    using vve = vv<VectorBuilder<ConcurrentType::STD_THREADS, threadCnt, seq, 1, ve>>;

//    using select_op = select_t<ve, less, uncompr_f, uncompr_f>;

    // @todo TOFIX: use physical partitioner for consolidation later
//    MetaOperator<vve, LogicalPartitioner, op2, LogicalPartitioner>::apply(baseCol1, uint64_t(3));

    std::cout << typestr(calc_binary_t<std::plus, ve, uncompr_f, uncompr_f, uncompr_f>::apply) << std::endl;

    auto presult
      = MetaOperator<
          vve,                                                           /// used (virtual) vector extension
          LogicalPartitioner,                                            /// Partitioner used for partitioning inputs
          calc_binary_t<std::plus, ve, uncompr_f, uncompr_f, uncompr_f>, /// virtualized executable/operator
          PhysicalPartitioner                                            /// Partitioner used for consolidating outputs
          >::apply(baseCol1, baseCol2);
    
    auto pbase1 = new PartitionedColumn<LogicalPartitioner, uncompr_f>(baseCol1, threadCnt);
    auto pbase2 = new PartitionedColumn<LogicalPartitioner, uncompr_f>(baseCol2, threadCnt);

    std::cout << "Output column:" << std::endl;
    for(uint64_t i = 0; i < threadCnt; ++i){
        std::cout << "Part" << i << ":" << std::endl;
        std::cout << std::setw(5) << "IN1" << " | " << std::setw(5) << "IN2" << " | " << std::setw(5) << "OUT" << " | " << std::endl;
        const column<uncompr_f> * pcol = (*presult)[i];
        const column<uncompr_f> * pb1 = (*pbase1)[i];
        const column<uncompr_f> * pb2 = (*pbase2)[i];
        print_columns(pb1, pb2, pcol);
    }
    
    using select_operator = MetaOperator<
      vve,
      LogicalPartitioner,
      select_t<ve, vectorlib::greater, uncompr_f, uncompr_f>,
      PhysicalPartitioner
    >;
    
//    std::cout << typestr(select_t<ve, vectorlib::less, uncompr_f, uncompr_f>::apply) << std::endl;
//    std::cout << typestr<select_operator::executableInputList_t>() << std::endl;
    auto presult_select
      = select_operator::apply(baseCol1, 70UL, 0UL);
    
    std::cout << "Output column:" << std::endl;
    for(uint64_t i = 0; i < threadCnt; ++i){
        std::cout << "Part" << i << ":" << std::endl;
        std::cout << "    " << std::setw(5) << "IN1" << " | " << std::setw(5) << "OUT" << " | " << std::endl;
        const column<uncompr_f> * pcol = (*presult_select)[i];
        const column<uncompr_f> * pb1 = (*pbase1)[i];
        print_columns(pb1, pcol);
    }
    

    return 0;
}
 
