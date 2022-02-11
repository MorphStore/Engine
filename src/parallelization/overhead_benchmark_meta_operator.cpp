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
 
#include <stdlibs>
#include <forward>

#include <vectorExtensions>
#include <virtual>
#include <monitor>
#include <utils>

#include <operators>
#include <storage>
#include <printing>
#include <core/utils/monitoring/CSVWriter.h>


#include <core/virtual/MetaOperator.h>
 
int main (){
    using namespace morphstore;
    using namespace vectorlib;
    using namespace virtuallib;
    //// ======= config ======= ////
    
    /// number of threads
    const uint64_t threadCnt = 14;
    
    /// numnber of test runs
    const uint64_t numberOfRuns = 10;
    
    /// print the results of ms::operator and metaOperator
    const bool printResults = false;
    
    /// N GB of data
    const size_t countValues = 1024 * 1024 / sizeof(uint64_t) * 1024 * 2;
    /// N KB of data
//    const size_t countValues = 1024 / sizeof(uint64_t) * 2;
    /// N values per thread
//    const size_t countValues = threadCnt * 10;
    
    
    /// Generate data
    std::cout << "Data generation of ";
    if(countValues * sizeof(uint64_t) >= 1024 * 1024 * 1024) {
        std::cout << ((float) countValues) / 1024 / 1024 / 1024 * sizeof(uint64_t) << " GB";
    } else if(countValues * sizeof(uint64_t) >= 1024 * 1024) {
        std::cout << ((float)countValues) / 1024 / 1024  * sizeof(uint64_t) << " MB";
    } else if(countValues * sizeof(uint64_t) >= 1024) {
        std::cout << ((float)countValues) / 1024   * sizeof(uint64_t) << " KB";
    } else {
        std::cout << ((float)countValues)  * sizeof(uint64_t) << " B";
    }
    std::cout << " (" << dotNumber(countValues) << " values) .." << std::flush;
    
    const column<uncompr_f> * const baseCol1
      = generate_with_distr(countValues, std::uniform_int_distribution<uint64_t>(0, 100), false, 8);
    std::cout << ".." << std::flush;
    const column<uncompr_f> * const baseCol2
      = generate_with_distr(countValues, std::uniform_int_distribution<uint64_t>(0, 100), false, 9);
    std::cout << " done." << std::endl;
    
    /// define vector extension
    using ve = scalar<v64<uint64_t>>;
    using vve = vv<VectorBuilder<ConcurrentType::STD_THREADS, threadCnt, seq, 1, ve>>;
    
    /// define operators
    using calc_op = calc_binary_t<std::plus, ve, uncompr_f, uncompr_f, uncompr_f>;
    using calc_op_meta
      = MetaOperator<
          vve,                               /// used (virtual) vector extension
          LogicalPartitioner,                /// Partitioner used for partitioning inputs
          calc_op,                           /// virtualized executable/operator
          PhysicalPartitioner                /// Partitioner used for consolidating outputs @todo
          >;
    
    using select_op = select_t<ve, vectorlib::greater, uncompr_f, uncompr_f>;
    using select_op_meta
      = MetaOperator<
          vve,
          LogicalPartitioner,
          select_op,
          PhysicalPartitioner
          >;
    
    
    using monitoring::MTask;
    monitoring::Monitor mon;
    
    std::cout << "Start Benchmark. Number of runs: 10" << std::endl;
    std::cout << std::endl;
    
    
    /// ====== Calc Operator ====== ///
    column<uncompr_f> const * calcResult = nullptr;
    /// PartitonedColumn<...> *
    calc_op_meta::output_t calcMetaResult = nullptr;
    
    /// run calc operator
    std::cout << "Run calc operator" << std::flush;
    for(uint64_t i = 0; i < numberOfRuns; ++i) {
        delete calcResult;
        mon.start(MTask::Test1);
        calcResult = calc_op::apply(baseCol1, baseCol2);
        mon.end(MTask::Test1);
        std::cout << "." << std::flush;
    }
    std::cout << "done." << std::endl;
    
    std::cout << "Execution time of calc operator: "
              << "mean: " << dotNumber(mon.getMean<std::chrono::nanoseconds>(MTask::Test1)) << " ns / "
              << "min:  " << dotNumber(mon.getMin<std::chrono::nanoseconds> (MTask::Test1)) << " ns / "
              << "max:  " << dotNumber(mon.getMax<std::chrono::nanoseconds> (MTask::Test1)) << " ns "
              << std::endl;
    
    
    /// run calc meta operator
    std::cout << "Run calc meta operator" << std::flush;
    for(uint64_t i = 0; i < numberOfRuns; ++i) {
        delete calcMetaResult;
        mon.start(MTask::Test2);
        calcMetaResult = calc_op_meta::apply(baseCol1, baseCol2);
        mon.end(MTask::Test2);
        std::cout << "." << std::flush;
    }
    std::cout << "done." << std::endl;
    
    std::cout << "Execution time of meta calc operator: "
              << "mean: " << dotNumber(mon.getMean<std::chrono::nanoseconds>(MTask::Test2)) << " ns / "
              << "min:  " << dotNumber(mon.getMin<std::chrono::nanoseconds> (MTask::Test2)) << " ns / "
              << "max:  " << dotNumber(mon.getMax<std::chrono::nanoseconds> (MTask::Test2)) << " ns "
              << std::endl;
    
    if constexpr(printResults) {
        auto calcResult_p = new PartitionedColumn<LogicalPartitioner, uncompr_f>(calcResult, threadCnt);
        std::cout << "Output column:" << std::endl;
        uint64_t offset = 0;
        for (uint64_t i = 0; i < threadCnt; ++ i) {
            std::cout << "Part " << i << ":" << std::endl;
            std::cout << "      " << std::setw(9) << "CALC" << " | " << std::setw(9) << "META CALC" << " | " << std::endl;
            const column<uncompr_f> * col1 = (*calcResult_p)[i];
            const column<uncompr_f> * col2 = (*calcMetaResult)[i];
            print_columns<9>(col1, col2);
        }
    }
    
    mon.reset(MTask::Test1);
    mon.reset(MTask::Test2);
    std::cout << std::endl;
   
    /// ====== Select Operator ====== ///
    column<uncompr_f> const * selectResult = nullptr;
    /// PartitonedColumn<...> *
    select_op_meta::output_t selectMetaResult = nullptr;
    
    
    /// run select operator
    std::cout << "Run select operator" << std::flush;
    for(uint64_t i = 0; i < numberOfRuns; ++i) {
        delete selectResult;
        mon.start(MTask::Test1);
        selectResult = select_op::apply(baseCol1, 70);
        mon.end(MTask::Test1);
        std::cout << "." << std::flush;
    }
    std::cout << "done." << std::endl;
    
    std::cout << "Execution time of select operator: "
              << "mean: " << dotNumber(mon.getMean<std::chrono::nanoseconds>(MTask::Test1)) << " ns / "
              << "min:  " << dotNumber(mon.getMin<std::chrono::nanoseconds> (MTask::Test1)) << " ns / "
              << "max:  " << dotNumber(mon.getMax<std::chrono::nanoseconds> (MTask::Test1)) << " ns "
              << std::endl;
    
    
    /// run select meta operator
    std::cout << "Run select meta operator" << std::flush;
    for(uint64_t i = 0; i < numberOfRuns; ++i) {
        delete selectMetaResult;
        mon.start(MTask::Test2);
        selectMetaResult = select_op_meta::apply(baseCol1, 70UL, 0UL);
        mon.end(MTask::Test2);
        std::cout << "." << std::flush;
    }
    std::cout << "done." << std::endl;
    
    std::cout << "Execution time of select meta operator: "
              << "mean: " << dotNumber(mon.getMean<std::chrono::nanoseconds>(MTask::Test2)) << " ns / "
              << "min:  " << dotNumber(mon.getMin<std::chrono::nanoseconds> (MTask::Test2)) << " ns / "
              << "max:  " << dotNumber(mon.getMax<std::chrono::nanoseconds> (MTask::Test2)) << " ns "
              << std::endl;
    
    
    
    
    if constexpr(printResults) {
        auto select_in = new PartitionedColumn<LogicalPartitioner, uncompr_f>(baseCol1, threadCnt);
        auto selectResult_p = new PartitionedColumn<LogicalPartitioner, uncompr_f>(selectResult, threadCnt);
        std::cout << "Output column:" << std::endl;
        uint64_t offset = 0;
        for (uint64_t i = 0; i < threadCnt; ++ i) {
            std::cout << "Part " << i << ":" << std::endl;
            std::cout << "      " << std::setw(11) << "SELECT" << " | " << std::setw(11) << "META SELECT" << " | " << std::endl;
            const column<uncompr_f> * col1 = (*select_in)[i];
            const column<uncompr_f> * col2 = (*selectResult_p)[i];
            const column<uncompr_f> * col3 = (*selectMetaResult)[i];
            print_columns<11>(offset, col1, col2, col3);
        }
    }
    
}
