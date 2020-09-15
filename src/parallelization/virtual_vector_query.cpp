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
 
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/project_uncompr.h>
#include <core/operators/general_vectorized/select_uncompr.h>
#include <core/operators/virtual_vectorized/select_uncompr.h>
#include <core/operators/general_vectorized/agg_sum_uncompr.h>
#include <core/operators/virtual_vectorized/agg_sum_uncompr.h>
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/storage/column.h>
#include <core/storage/column_gen.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>

#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

#include <core/utils/monitoring/Monitor.h>
#include <core/utils/system.h>

#include <functional>
#include <iostream>
#include <random>

using namespace morphstore;
using namespace vectorlib;

// ****************************************************************************
// * Example query
// ****************************************************************************

// SELECT SUM(baseCol2) WHERE baseCol1 = 150

int main( void ) {
    // ************************************************************************
    // * Generation of the synthetic base data
    // ************************************************************************
    
    std::cout << "Base data generation started... ";
    std::cout.flush();
    
    /// 2 GB of data
//    const size_t countValues = 2 * 1024 * 1024 / sizeof(uint64_t) * 1024; // todo : 200gb
//    const size_t countValues = 2 * 1024 * 1024 / sizeof(uint64_t) ; /// 2 MB
    const size_t countValues = 1024; /// 8192 B
//    const size_t countValues = 20;
    const column<uncompr_f> * const  baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(100, 199),
            false,
            8
    );
    const column<uncompr_f> * const  baseCol2 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(0, 10),
            false,
            42
    );
    const column<uncompr_f> * const  baseCol3 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(0, 10),
            false,
            21
    );
    
    
//	print_columns(print_buffer_base::decimal, baseCol1, "baseCol1");
//	print_columns(print_buffer_base::decimal, baseCol2, "baseCol2");
    
    std::cout << "done." << std::endl;
    
    // ************************************************************************
    // * Query execution
    // ************************************************************************
    
    std::string benchname;
    using ve2 = avx2<v256<uint64_t> >;
    
//    benchname = "avx512";
//    using ve = avx512<v512<uint64_t> >;

//    benchname = "avx2";
//    using ve = avx2<v256<uint64_t> >;

//    benchname = "sse_128";
//    using ve = sse<v128<uint64_t> >;

    benchname = "scalar_select";
    using ve = scalar<v64<uint64_t> >;

    
    
//    using vve = vv< v8192<uint64_t>, avx512<v512<uint64_t>> >;
//    using vve = vv< v2048<uint64_t>, avx512<v512<uint64_t>> >;
    
    std::ofstream file;
    testAndCreateDirectory("./output/thread_spawning/");
    file.open("./output/thread_spawning/" + benchname + ".csv");
	
	#define RUNS 10
 
//    using vve = vv< v512<uint64_t>, avx512<v512<uint64_t>> >;
    
    std::cout << "Query execution started... ";
    std::cout.flush();
    
    // Positions fulfilling "baseCol1 = 150"
//    auto i1 = my_select_wit_t<
//            equal,
//            ve,
//            uncompr_f,
//            uncompr_f
//    >::apply(baseCol1, 150);
    // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
//    auto i2 = my_project_wit_t<
//            ve,
//            uncompr_f,
//            uncompr_f,
//            uncompr_f
//    >::apply(baseCol2, i1);
    // Sum over the data elements of "baseCol2" fulfilling "baseCol1 = 150"
//    auto i3 = agg_sum<vve, uncompr_f>(i2);


//    auto i3 = agg_sum<vve, uncompr_f>(i2);
	using namespace monitoring;
	monitoring::Monitor monBase;
 
	
	column<uncompr_f> const * out1[RUNS];
	column<uncompr_f> const * out2[RUNS];
	column<uncompr_f> const * out3[RUNS];
	
	std::cout << std::endl;
    /// Base test
    
    
//	for (int i = 0; i < runs; ++i) {
//		auto begin =  clock::now();
//		volatile auto col2a = agg_sum<ve, uncompr_f>(baseCol2);
//		auto end = clock::now();
//		volatile auto col2b = agg_sum<ve, uncompr_f>(baseCol3);
//
//		std::cout << "Run" << i << " [2nd]: " << std::chrono::duration_cast<nanoseconds>(end - begin).count() << std::endl;
//
//        print_columns(print_buffer_base::decimal, col2a, "SUM(baseCol2)");
//        print_columns(print_buffer_base::decimal, col2b, "SUM(baseCol2)");
//
////		std::cout << "Run" << i << " : " << monBase.getDuration<nanoseconds>(MTask::Runtime) << std::endl;
//	}
 
	for (int i = 0; i < RUNS; ++i) {
		monBase.start(MTask::Runtime);
		
		out1[i] = agg_sum<ve, uncompr_f>(baseCol1);
		monBase.end(MTask::Runtime);
		std::cout << "Run" << i << " [1st]: " << monBase.getDuration<nanoseconds>(MTask::Runtime) << std::endl;
	}
	
	
	std::cout << "Average duration of agg_sum with " << benchname << ": "
	          << monBase.getAvg<milliseconds>(MTask::Runtime) << " ms "
	          << "min: " << monBase.getMin<milliseconds>(MTask::Runtime) << " ms "
	          << "max: " << monBase.getMax<milliseconds>(MTask::Runtime) << " ms "
	          << std::endl;
	
	file << benchname << "<" << ve::vector_helper_t::size_bit::value << ">" << ";";
	for(auto time : monBase.getAllDurations<nanoseconds>(MTask::Runtime)){
		file << time << ";";
	}
	file << "\n";
    
	
    /// Test with 64bit vector
    if(ve::vector_helper_t::size_bit::value <= 64) {
    	using virt = vv<v64<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			out3[i] = select_t<virt, equal>::apply(baseCol2, 5);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }
    
    
	for (int j = 0; j < RUNS; ++j) {
//        print_columns(print_buffer_base::decimal, out1[j], "Reference SUM(baseCol1)");
//        print_columns(print_buffer_base::decimal, out2[j], "Virtual SUM(baseCol1)");
        print_columns(print_buffer_base::decimal, baseCol2, "baseCol2)");
        print_columns(print_buffer_base::decimal, out3[j], "Virtual SELECT baseCol2 > 5)");
		return 0;
	}

    if(ve::vector_helper_t::size_bit::value <= 128) {
    	using virt = vv<v128<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 256) {
    	using virt = vv<v256<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 512) {
    	using virt = vv<v512<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 1024) {
    	using virt = vv<v1024<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 2048) {
    	using virt = vv<v2048<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 4096) {
    	using virt = vv<v4096<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 8192) {
    	using virt = vv<v8192<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 16384) {
    	using virt = vv<v16384<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
			out2[i] = agg_sum<virt, uncompr_f>(baseCol1);
			mon.end(MTask::Runtime);
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
		          << mon.getAvg<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }
	
	file.close();
    
    std::cout << "done." << std::endl << std::endl;
    
    // ************************************************************************
    // * Result output
    // ************************************************************************
	
	for (int j = 0; j < RUNS; ++j) {
        print_columns(print_buffer_base::decimal, out1[j], "Reference SUM(baseCol2)");
        print_columns(print_buffer_base::decimal, out2[j], "Virtual SUM(baseCol2)");
	}
//    info("Virtual Vector:");
//    print_columns(print_buffer_base::decimal, out2, "SUM(baseCol2)");
    
    return 0;
}
