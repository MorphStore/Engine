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
 
/// basics
#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/morphing/uncompr.h>

/// operators
#include <core/operators/uncompr/project.h>
#include <core/operators/otfly_derecompr/select.h>
#include <core/operators/uncompr/select.h>
#include <core/operators/uncompr/agg_sum_all.h>
#include <core/operators/otfly_derecompr/agg_sum_all.h>
#include <core/operators/virtual_vectorized/select_uncompr.h>
#include <core/operators/virtual_vectorized/project_uncompr.h>
#include <core/operators/virtual_vectorized/agg_sum_all.h>

/// storage
#include <core/storage/column.h>
#include <core/storage/column_gen.h>

/// utils
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>

/// primitives
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>

/// monitoring
#include <core/utils/monitoring/Monitor.h>
#include <core/utils/system.h>

/// std libs
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
    
    /// todo : 200gb
    /// 2 MB
//    const size_t countValues = 1024 * 1024 / sizeof(uint64_t) * 2;
    /// 200 MB
//    const size_t countValues = 1024 * 1024 / sizeof(uint64_t) * 2 * 100;
    /// 2 GB of data
    const size_t countValues = 1024 * 1024 / sizeof(uint64_t) * 2 * 1024;

    /// 8192 B
//    const size_t countValues = 1024;
//    const size_t countValues = 256 + 3;
//    const size_t countValues = 20;
    const column<uncompr_f> * const  baseCol1 = generate_with_distr(
            countValues,
            std::uniform_int_distribution<uint64_t>(0, 100),
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
    
/// ************************************************************************
/// * Query execution
/// ************************************************************************
    
    std::string benchname;
    using ve2 = avx2<v256<uint64_t> >;
    
//    benchname = "avx512";
//    #define VECTOR_BITSIZE 512
//    using ve = avx512<vector_view<VECTOR_BITSIZE, uint64_t> >;

//    benchname = "avx2";
//    #define VECTOR_BITSIZE 256
//    using ve = avx2<vector_view<VECTOR_BITSIZE, uint64_t> >;

//    benchname = "sse_128";
//    #define VECTOR_BITSIZE 128
//    using ve = sse<vector_view<VECTOR_BITSIZE, uint64_t> >;

    benchname = "scalar";
    #define VECTOR_BITSIZE 64
    using ve = scalar<vector_view<VECTOR_BITSIZE, uint64_t> >;

    
    
//    using vve = vv< v8192<uint64_t>, avx512<v512<uint64_t>> >;
//    using vve = vv< v2048<uint64_t>, avx512<v512<uint64_t>> >;
    
    /// open output file (overwrite)
    std::ofstream file;
    testAndCreateDirectory("./output/thread_spawning/");
    file.open("./output/thread_spawning/" + benchname + ".csv");
	
    /// number of test runs
	#define RUNS 10
 
//    using vve = vv< v512<uint64_t>, avx512<v512<uint64_t>> >;
    
    std::cout << "Query execution started... " << std::endl;
    
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
	
	column<uncompr_f> const * test[10];
	
	std::cout << std::endl;
    /// Base test
	for (int i = 0; i < RUNS; ++i) {
		monBase.start(MTask::Runtime);
		/// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
		auto positions = select<ve, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
		auto filteredData = project<ve, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
		out1[i] = agg_sum_all<ve, uncompr_f>(filteredData);
		monBase.end(MTask::Runtime);
		delete positions;
		delete filteredData;
//		std::cout << "Run" << i << " [1st]: " << monBase.getDuration<nanoseconds>(MTask::Runtime) << std::endl;
	}
	
	
	std::cout << "Average duration of agg_sum with " << benchname << ": "
              << monBase.getMean<milliseconds>(MTask::Runtime) << " ms "
	          << "min: " << monBase.getMin<milliseconds>(MTask::Runtime) << " ms "
	          << "max: " << monBase.getMax<milliseconds>(MTask::Runtime) << " ms "
	          << std::endl;
	
	std::cout << "Size of columns {";
	for(int i = 0; i < RUNS; ++i){
	    std::cout << out1[i]->get_count_values() << ", ";
	    delete out1[i];
	}
	std::cout << "}" << std::endl;
	
	file << benchname << "<" << ve::vector_helper_t::size_bit::value << ">" << ";";
	for(auto time : monBase.getAllDurations<nanoseconds>(MTask::Runtime)){
		file << time << ";";
	}
	file << "\n";
 
	
	
	/// ============================================================================================================ ///
	/// === Test different vector lengths ========================================================================== ///
	
	
    /// Test with 64bit vector
    #if VECTOR_BITSIZE <= 64
    if constexpr(ve::vector_helper_t::size_bit::value <= 64) {
    	using virt = vv_old<v64<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;
        
		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }
    #endif
    
    
	for (int j = 0; j < RUNS; ++j) {
//        print_columns(print_buffer_base::decimal, out1[j], "Reference SUM(baseCol1)");
//        print_columns(print_buffer_base::decimal, out2[j], "Virtual SUM(baseCol1)");
//        print_columns(print_buffer_base::decimal, baseCol2, "baseCol2");
//        print_columns(print_buffer_base::decimal, out3[j], "Virtual SELECT baseCol2 > 5)");
	}
    
    #if VECTOR_BITSIZE <= 128
    if(ve::vector_helper_t::size_bit::value <= 128) {
    	using virt = vv_old<v128<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }
    #endif


    #if VECTOR_BITSIZE <= 256
    if(ve::vector_helper_t::size_bit::value <= 256) {
    	using virt = vv_old<v256<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }
    #endif

    if(ve::vector_helper_t::size_bit::value <= 512) {
    	using virt = vv_old<v512<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 1024) {
    	using virt = vv_old<v1024<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 2048) {
    	using virt = vv_old<v2048<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 4096) {
    	using virt = vv_old<v4096<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 8192) {
    	using virt = vv_old<v8192<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

		file << "vv" << "<" << virt::vector_helper_t::size_bit::value << ">" << ";";
		for(auto time : mon.getAllDurations<nanoseconds>(MTask::Runtime)){
			file << time << ";";
		}
		file << "\n";
    }


    if(ve::vector_helper_t::size_bit::value <= 16384) {
    	using virt = vv_old<v16384<uint64_t>, ve>;
    	Monitor mon;
		for (int i = 0; i < RUNS; ++i) {
			mon.start(MTask::Runtime);
            /// Query: SELECT SUM(baseCol1) FROM someTable WHERE baseCol1 < 80
            auto positions = select<virt, vectorlib::less, uncompr_f, uncompr_f>(baseCol1, 80, 0);
            auto filteredData = project<virt, uncompr_f, uncompr_f, uncompr_f>(baseCol1, positions);
            out1[i] = agg_sum_all<virt, uncompr_f>(filteredData);
			mon.end(MTask::Runtime);
            delete positions;
            delete filteredData;
		}
		std::cout << "Average duration of agg_sum with vv<" << virt::vector_helper_t::size_bit::value << ">: "
                  << mon.getMean<milliseconds>(MTask::Runtime) << " ms "
		          << "min: " << mon.getMin<milliseconds>(MTask::Runtime) << " ms "
		          << "max: " << mon.getMax<milliseconds>(MTask::Runtime) << " ms "
		          << std::endl;

        std::cout << "Size of columns {";
        for(int i = 0; i < RUNS; ++i){
            std::cout << out1[i]->get_count_values() << ", ";
            delete out1[i];
        }
        std::cout << "}" << std::endl;

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
//        print_columns(print_buffer_base::decimal, out1[j], "Reference SUM(baseCol2)");
//        print_columns(print_buffer_base::decimal, out2[j], "Virtual SUM(baseCol2)");
	}
//    info("Virtual Vector:");
//    print_columns(print_buffer_base::decimal, out2, "SUM(baseCol2)");
    
    return 0;
}
