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





struct op : Executable {
    static
    uint64_t apply(column<uncompr_f> * const, uint64_t * ){
        std::cout << "called op::apply()" << std::endl;
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
    const size_t countValues = sizeof(uint64_t) * 2 * 1024;

    column<uncompr_f> * const baseCol1 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                8
            )
          )
      );
    column<uncompr_f> * const baseCol2 =
       reinterpret_cast< column<uncompr_f> * >(
          const_cast<column<uncompr_f> * >(
            generate_with_distr(
                countValues,
                std::uniform_int_distribution<uint64_t>(0, 100),
                false,
                9
            )
          )
      );


    using ve = scalar<v64<uint64_t>>;
    using vve = vv<VectorBuilder<ConcurrentType::STD_THREADS, 4, seq, 1, ve>>;

//    using select_op = select_t<ve, less, uncompr_f, uncompr_f>;
//
//    auto partitions = logical_partitioning<vve::base_t>
//      ::apply(baseCol1->get_data(), countValues, vve::vectorBuilder::cvalue::value);
//    /// less than 5
//    auto comparatorValue = new std::vector<uint64_t>(vve::vectorBuilder::cvalue::value, 5);
//
//    std::cout << "Pointer: " << (uint64_t*) baseCol1->get_data() << std::endl;
//    PartitionedColumn<LogicalPartitioner, uncompr_f, uint64_t> PC(baseCol1,4);
    
//    std::cout << "input type: " << typestr<unfold_type(op2::apply)::inputType>() << std::endl;
//    std::cout << "output type: " << typestr<unfold_type(op2::apply)::returnType>() << std::endl;
//    std::cout << "pack type: " << typestr<pack< typename unfold_type(op2::apply)::returnType >::type>() << std::endl;

    // @todo TOFIX: use physical partitioner for consolidation later
//    MetaOperator<vve, LogicalPartitioner, op2, LogicalPartitioner>::apply(baseCol1, uint64_t(3));

    std::cout << typestr(calc_binary_t<std::plus, ve, uncompr_f, uncompr_f, uncompr_f>::apply) << std::endl;

    MetaOperator<vve, LogicalPartitioner, calc_binary_t<std::plus, ve, uncompr_f, uncompr_f, uncompr_f>, LogicalPartitioner>::apply(baseCol1, baseCol2);
//    std::cout << typestr<decltype(MetaOperator<vve, LogicalPartitioner, op2, LogicalPartitioner>::apply<column<uncompr_f>*, uint64_t>)>() << std::endl;
    
    return 0;
    
    
    
    using function = uint64_t (uint64_t, uint64_t);
    using function2 = uint64_t (*)(uint64_t, uint64_t);
    using function3 = decltype(op::apply);
    using function4 = uint64_t (uint64_t, uint64_t)&;
    
    function f1;
    function2 f2;
    function3 f3;

//    f2 = op::apply;
    
    uint64_t a(3);
    uint64_t& b = a;
    uint64_t * c = &a;
    
    std::cout << "1 " << typestr<function>() << std::endl;
    std::cout << "1 " << typestr<decltype(f1)>() << std::endl;
    std::cout << "2 " << typestr<function2>() << std::endl;
    std::cout << "2 " << typestr<decltype(f2)>() << std::endl;
    std::cout << "2 " << typestr(f2) << std::endl;
    std::cout << "3 " << typestr<function3>() << std::endl;
    std::cout << "3 " << typestr<decltype(f3)>() << std::endl;
//    std::cout << "3 " << typestr(f3) << std::endl;
//    std::cout << "4 " << typestr<function4>() << std::endl;
    std::cout << "40 " << typestr<decltype(op::apply)>() << std::endl;
    std::cout << "5 " << typestr<decltype(&op::apply)>() << std::endl;
    std::cout << "6 " << typestr(op::apply) << std::endl;
    std::cout << "7 " << typestr<unfold_type(op::apply)::type>() << std::endl;

    std::cout << "8 " << typestr(a) << std::endl;
    std::cout << "9 " << typestr(b) << std::endl;
    std::cout << "10 " << typestr(uint64_t(3)) << std::endl;
    std::cout << "11 " << typestr(c) << std::endl;
    std::cout << "11 " << typestr<uint64_t*>(c) << std::endl;
//    std::cout << "11 " << typestr(decltype(c)(c)) << std::endl;
    
    std::cout << is_storage<column<uncompr_f>*>::value << std::endl;

}
 
