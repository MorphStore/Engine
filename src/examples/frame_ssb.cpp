/**********************************************************************************************
 * Copyright (C) 2020 by MorphStore-Team                                                      *
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

/**
 * @brief This file contains experimental setting for the evaluation of parall
 * queries on replicated columns via abstraction layer
 */

#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"
#include <core/operators/general_vectorized/merge_uncompr.h>
#include <core/operators/general_vectorized/group_compr.h>
#include <core/operators/general_vectorized/group_uncompr.h>
#include "../../include/core/operators/scalar/agg_sum_uncompr.h"
#include "../../include/core/operators/scalar/project_uncompr.h"
#include "../../include/core/operators/scalar/select_uncompr.h"
#include "../../include/core/storage/column.h"
#include "../../include/core/storage/column_gen.h"
#include "../../include/core/utils/basic_types.h"
#include "../../include/core/utils/printing.h"
#include "../../include/vector/scalar/extension_scalar.h"
#include <functional>
#include <iostream>
#include <random>

#include <core/memory/mm_glob.h>
#include <core/morphing/format.h>
#include <core/persistence/binary_io.h>
#include <core/storage/column.h>
#include <core/utils/basic_types.h>
#include <core/utils/printing.h>
#include <vector/vector_extension_structs.h>
#include <vector/vector_primitives.h>
#include <core/morphing/uncompr.h>
#include <core/operators/general_vectorized/agg_sum_compr.h>
#include <core/operators/general_vectorized/calc_uncompr.h>
#include <core/operators/general_vectorized/intersect_uncompr.h>
#include <core/operators/general_vectorized/join_compr.h>
#include <core/operators/general_vectorized/project_compr.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/utils/column_info.h>
#include <core/storage/column_gen.h>
#include <core/utils/histogram.h>
#include <core/utils/monitoring.h>
#include <vector/primitives/compare.h>
#include <functional>
#include <iostream>
#include <vector>
#include <numa.h>
#include <pthread.h>
#include <thread>
#include <atomic>
#include <core/operators/scalar/basic_operators.h>
#include <core/operators/scalar/append.h>
#include <core/operators/general_vectorized/select_compr.h>
#include <core/storage/replicated_column.h>
#include <abstract/abstract_layer.h>
//#include <abstract/memory_counter.h>
//#include <abstract/numa_counter.h>
#include <abstract/stopwatch.h>

using namespace vectorlib;
using namespace morphstore;

// ****************************************************************************
// Multi-threading related global variables
// ****************************************************************************

// Number of concurrent query workers
#ifdef SSB_COUNT_THREADS
size_t countThreads = SSB_COUNT_THREADS;
#else
size_t countThreads = 1;
#endif

// Test runtime
#ifdef SSB_RUNTIME
size_t measuringTimeMillis = SSB_RUNTIME;
#else
size_t measuringTimeMillis = 10000;
#endif

// Number of various queries
#define QUERIES_NUM 13

// PCM counters
//PCM* m;

std::atomic<size_t> waitingThreads; // = {countThreads};

struct ThreadData {
    pthread_t thread;
    size_t node = 0;
    size_t core = 0;
    size_t nodeSize = 0;
    bool halt = false;

    size_t runsExecuted = 0;
    WallClockStopWatch sw;

    size_t queryNum = 0;
};

// ****************************************************************************
// Base data declaration
// ****************************************************************************

// Replicated base data.
struct replicated_customer_t {
    replicated_column * c_city;
    replicated_column * c_custkey;
    replicated_column * c_nation;
    replicated_column * c_region;
} replicated_customer;
struct replicated_date_t {
    replicated_column * d_datekey;
    replicated_column * d_weeknuminyear;
    replicated_column * d_year;
    replicated_column * d_yearmonth;
    replicated_column * d_yearmonthnum;
} replicated_date;
struct replicated_lineorder_t {
    replicated_column * lo_custkey;
    replicated_column * lo_discount;
    replicated_column * lo_extendedprice;
    replicated_column * lo_orderdate;
    replicated_column * lo_partkey;
    replicated_column * lo_quantity;
    replicated_column * lo_revenue;
    replicated_column * lo_suppkey;
    replicated_column * lo_supplycost;
} replicated_lineorder;
struct replicated_part_t {
    replicated_column * p_brand;
    replicated_column * p_category;
    replicated_column * p_mfgr;
    replicated_column * p_partkey;
} replicated_part;
struct replicated_supplier_t {
    replicated_column * s_city;
    replicated_column * s_nation;
    replicated_column * s_region;
    replicated_column * s_suppkey;
} replicated_supplier;

// Normal base data.
struct customer_t {
    const column<uncompr_f> * c_city;
    const column<uncompr_f> * c_custkey;
    const column<uncompr_f> * c_nation;
    const column<uncompr_f> * c_region;
} customer;
struct date_t {
    const column<uncompr_f> * d_datekey;
    const column<uncompr_f> * d_weeknuminyear;
    const column<uncompr_f> * d_year;
    const column<uncompr_f> * d_yearmonth;
    const column<uncompr_f> * d_yearmonthnum;
} date;
struct lineorder_t {
    const column<uncompr_f> * lo_custkey;
    const column<uncompr_f> * lo_discount;
    const column<uncompr_f> * lo_extendedprice;
    const column<uncompr_f> * lo_orderdate;
    const column<uncompr_f> * lo_partkey;
    const column<uncompr_f> * lo_quantity;
    const column<uncompr_f> * lo_revenue;
    const column<uncompr_f> * lo_suppkey;
    const column<uncompr_f> * lo_supplycost;
} lineorder;
struct part_t {
    const column<uncompr_f> * p_brand;
    const column<uncompr_f> * p_category;
    const column<uncompr_f> * p_mfgr;
    const column<uncompr_f> * p_partkey;
} part;
struct supplier_t {
    const column<uncompr_f> * s_city;
    const column<uncompr_f> * s_nation;
    const column<uncompr_f> * s_region;
    const column<uncompr_f> * s_suppkey;
} supplier;


size_t dataCount;
user_params setting;

// ****************************************************************************
// Intel PCM monitoring thread function
// ****************************************************************************
/*
void* updater(void* parameters)
{
    ThreadData* td = (ThreadData*) parameters;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    if (td->core < td->nodeSize || td->core >= td->nodeSize*3)
        CPU_SET(td->core, &cpuset);
    else if (td->core < td->nodeSize*2)
        CPU_SET(td->core + td->nodeSize, &cpuset);
    else
        CPU_SET(td->core - td->nodeSize, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    numa_set_preferred(td->node);

    waitingThreads--;

    while (waitingThreads > 0)
    {
       //wait for synchronized start
    }

    set_signal_handlers();

/// NUMA

    bool show_core_output = true;
    bool show_partial_core_output = false;
    bool show_socket_output = true;
    bool show_system_output = true;
    bool csv_output = false;
    bool reset_pmu = false;
    bool allow_multiple_instances = false;
    bool disable_JKT_workaround = false; // as per http://software.intel.com/en-us/articles/performance-impact-when-sampling-certain-llc-events-on-snb-ep-with-vtune

    std::bitset<MAX_CORES> ycores;
///
    double delay = -1.0;
    bool csv = false, csvheader=false, show_channel_output=true;
    uint32 no_columns = DEFAULT_DISPLAY_COLUMNS; // Default number of columns is 2

    char * sysCmd = NULL;
    char ** sysArgv = NULL;

#ifndef _MSC_VER
    long diff_usec = 0; // deviation of clock is useconds between measurements
    int calibrated = PCM_CALIBRATION_INTERVAL - 2; // keeps track is the clock calibration needed
#endif
    int rankA = -1, rankB = -1;
    bool PMM = true;
    unsigned int numberOfIterations = 0; // number of iterations ?

///
    std::vector<CoreCounterState> cstates1, cstates2;
    std::vector<SocketCounterState> sktstate1, sktstate2;
    SystemCounterState sstate1, sstate2;
    const auto cpu_model = m->getCPUModel();
    uint64 TimeAfterSleep = 0;
    PCM_UNUSED(TimeAfterSleep);

    if (delay <= 0.0) delay = PCM_DELAY_DEFAULT;
///

    m->disableJKTWorkaround();
    print_cpu_details();
    if (!m->hasPCICFGUncore())
    {
        std::cerr << "Unsupported processor model (" << m->getCPUModel() << ")." << std::endl;
        if (m->memoryTrafficMetricsAvailable())
            cerr << "For processor-level memory bandwidth statistics please use pcm.x" << endl;
        exit(EXIT_FAILURE);
    }

    if(PMM && (m->PMMTrafficMetricsAvailable() == false))
    {
        cerr << "PMM traffic metrics are not available on your processor." << endl;
        exit(EXIT_FAILURE);
    }

    if((rankA >= 0 || rankB >= 0) && PMM)
    {
        cerr << "PMM traffic metrics are not available on rank level" << endl;
        exit(EXIT_FAILURE);
    }

    if((rankA >= 0 || rankB >= 0) && !show_channel_output)
    {
        cerr << "Rank level output requires channel output" << endl;
        exit(EXIT_FAILURE);
    }

    //PCM::ErrorCode status = m->program();
    //PCM::ErrorCode status2 = m->programServerUncoreMemoryMetrics(rankA, rankB, PMM);
/*
    switch (status)
    {
        case PCM::Success:
            break;
        case PCM::MSRAccessDenied:
            cerr << "Access to Processor Counter Monitor has denied (no MSR or PCI CFG space access)." << endl;
            exit(EXIT_FAILURE);
        case PCM::PMUBusy:
            cerr << "Access to Processor Counter Monitor has denied (Performance Monitoring Unit is occupied by other application). Try to stop the application that uses PMU." << endl;
            cerr << "Alternatively you can try to reset PMU configuration at your own risk. Try to reset? (y/n)" << endl;
            char yn;
            std::cin >> yn;
            if ('y' == yn)
            {
                m->resetPMU();
                cerr << "PMU configuration has been reset. Try to rerun the program again." << endl;
            }
            exit(EXIT_FAILURE);
        default:
            cerr << "Access to Processor Counter Monitor has denied (Unknown error)." << endl;
            exit(EXIT_FAILURE);
    }

    switch (status2)
    {
        case PCM::Success:
            break;
        case PCM::MSRAccessDenied:
            cerr << "Access to Processor Counter Monitor has denied (no MSR or PCI CFG space access)." << endl;
            exit(EXIT_FAILURE);
        case PCM::PMUBusy:
            cerr << "Access to Processor Counter Monitor has denied (Performance Monitoring Unit is occupied by other application). Try to stop the application that uses PMU." << endl;
            cerr << "Alternatively you can try to reset PMU configuration at your own risk. Try to reset? (y/n)" << endl;
            char yn;
            std::cin >> yn;
            if ('y' == yn)
            {
                m->resetPMU();
                cerr << "PMU configuration has been reset. Try to rerun the program again." << endl;
            }
            exit(EXIT_FAILURE);
        default:
            cerr << "Access to Processor Counter Monitor has denied (Unknown error)." << endl;
            exit(EXIT_FAILURE);
    }
*//*

    if(m->getNumSockets() > max_sockets)
    {
        cerr << "Only systems with up to "<<max_sockets<<" sockets are supported! Program aborted" << endl;
        exit(EXIT_FAILURE);
    }

    ServerUncorePowerState * BeforeState = new ServerUncorePowerState[m->getNumSockets()];
    ServerUncorePowerState * AfterState = new ServerUncorePowerState[m->getNumSockets()];
    uint64 BeforeTime = 0, AfterTime = 0;

    //if ( (sysCmd != NULL) && (delay<=0.0) ) {
        // in case external command is provided in command line, and
        // delay either not provided (-1) or is zero
    //m->setBlocked(true);
    //} else {
        m->setBlocked(false);
    //}

    if (csv) {
        if( delay<=0.0 ) delay = PCM_DELAY_DEFAULT;
    } else {
        // for non-CSV mode delay < 1.0 does not make a lot of practical sense: 
        // hard to read from the screen, or
        // in case delay is not provided in command line => set default
        if( ((delay<1.0) && (delay>0.0)) || (delay<=0.0) ) delay = PCM_DELAY_DEFAULT;
    }

    cerr << "Update every "<<delay<<" seconds"<< endl;

    while (!td->halt)
    {
    for(uint32 i=0; i<m->getNumSockets(); ++i)
        BeforeState[i] = m->getServerUncorePowerState(i); 

    BeforeTime = m->getTickCount();

    if( sysCmd != NULL ) {
        MySystem(sysCmd, sysArgv);
    }
///
    m->getAllCounterStates(sstate1, sktstate1, cstates1);
///
    unsigned int i = 1;

    //while ((i <= numberOfIterations) || (numberOfIterations == 0))
    {
        if(!csv) cout << std::flush;
        int delay_ms = int(delay * 1000);
        int calibrated_delay_ms = delay_ms;

        // compensation of delay on Linux/UNIX
        // to make the samling interval as monotone as possible
        struct timeval start_ts, end_ts;
        if(calibrated == 0) {
            gettimeofday(&end_ts, NULL);
            diff_usec = (end_ts.tv_sec-start_ts.tv_sec)*1000000.0+(end_ts.tv_usec-start_ts.tv_usec);
            calibrated_delay_ms = delay_ms - diff_usec/1000.0;
        }

        MySleepMs(calibrated_delay_ms);

#ifndef _MSC_VER
        calibrated = (calibrated + 1) % PCM_CALIBRATION_INTERVAL;
        if(calibrated == 0) {
            gettimeofday(&start_ts, NULL);
        }
#endif

        AfterTime = m->getTickCount();
        for(uint32 i=0; i<m->getNumSockets(); ++i)
            AfterState[i] = m->getServerUncorePowerState(i);

    if (!csv) {
      //cout << "Time elapsed: "<<dec<<fixed<<AfterTime-BeforeTime<<" ms\n";
      //cout << "Called sleep function for "<<dec<<fixed<<delay_ms<<" ms\n";
    }

        if(rankA >= 0 || rankB >= 0)
          calculate_bandwidth(m,BeforeState,AfterState,AfterTime-BeforeTime,csv,csvheader, no_columns, rankA, rankB);
        else
          calculate_bandwidth(m,BeforeState,AfterState,AfterTime-BeforeTime,csv,csvheader, no_columns, PMM, show_channel_output);

///
        //cerr << "debug: " << m->getNumSockets() << "  " << m->incomingQPITrafficMetricsAvailable() << endl;
        TimeAfterSleep = m->getTickCount();
        m->getAllCounterStates(sstate2, sktstate2, cstates2);

        //if (csv_output)
        //    print_csv(m, cstates1, cstates2, sktstate1, sktstate2, ycores, sstate1, sstate2,
        //    cpu_model, show_core_output, show_partial_core_output, show_socket_output, show_system_output);
        //else
        //{
            print_output(m, cstates1, cstates2, sktstate1, sktstate2, ycores, sstate1, sstate2,
            cpu_model, show_core_output, show_partial_core_output, show_socket_output, show_system_output);
        //}
        swap(sstate1, sstate2);
        swap(sktstate1, sktstate2);
        swap(cstates1, cstates2);
///
        swap(BeforeTime, AfterTime);
        swap(BeforeState, AfterState);

        if ( m->isBlocked() ) {
        // in case PCM was blocked after spawning child application: break monitoring loop here
            break;
        }
    ++i;
    }
    }

    delete[] BeforeState;
    delete[] AfterState;

    return nullptr;
}*/

// ****************************************************************************
// Queries executed by thread workers
// ****************************************************************************

std::atomic<size_t> counters[QUERIES_NUM] = {};
std::vector<void (*)(ThreadData*)> queries;

void query_q11(ThreadData* td);
void query_q12(ThreadData* td);
void query_q13(ThreadData* td);
void query_q21(ThreadData* td);
void query_q22(ThreadData* td);
void query_q23(ThreadData* td);
void query_q31(ThreadData* td);
void query_q32(ThreadData* td);
void query_q33(ThreadData* td);
void query_q34(ThreadData* td);
void query_q41(ThreadData* td);
void query_q42(ThreadData* td);
void query_q43(ThreadData* td);

void* query_executor(void* parameters)
{
    // ************************************************************************
    // * Preparation of the affiliation setup
    // ************************************************************************
    ThreadData* td = (ThreadData*) parameters;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    uint32_t cpu;

    // First use native cores, then hyper cores, then next node is started. 2 sockets.
//    if (td->core < td->nodeSize || td->core >= td->nodeSize * 3)
//        CPU_SET(td->core, &cpuset);
//    else if (td->core < td->nodeSize*2)
//        CPU_SET(td->core + td->nodeSize, &cpuset);
//    else
//        CPU_SET(td->core - td->nodeSize, &cpuset);
    if (td->core < 3)
      CPU_SET(td->core, &cpuset);
    else
      CPU_SET(td->core + td->nodeSize - 3, &cpuset);
    // First use native cores, then hyper cores, then next node is started. 2 sockets.
//    if (td->core < td->nodeSize)
//        CPU_SET(td->core, &cpuset);
//    else
//        CPU_SET(td->core + td->nodeSize, &cpuset);

    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    // Ensure local intermidiate results allocation
    cpu = sched_getcpu();
    td->node = numa_node_of_cpu(cpu);
    numa_set_preferred(td->node);

    // Random device to obtain a seed for the random number engine
    std::random_device rd;
    // Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, QUERIES_NUM-1);

    // Wait for synchronized start
    waitingThreads--;
    while (waitingThreads > 0)
    {
    }

    // Desynchronized start
    //std::this_thread::sleep_for(std::chrono::milliseconds((dis(gen)*100) + ((dis(gen))+5)*100));

    td->sw.start();

    while (!td->halt)
    {
       // Pick up randomly one query for execution
       size_t random = dis(gen); //td->queryNum;
       queries[random](td);
       counters[random]++;
       td->runsExecuted++;
    }

    td->sw.stop();

    return nullptr;
}

void query_q11(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_discount_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_discount);
    auto replicated_lineorder_lo_extendedprice_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_extendedprice);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_quantity_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_quantity);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_47_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_lineorder.lo_discount, 1);
    auto C_47_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_lineorder.lo_discount, 3, 21810975);
    auto C_47 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_47_lo, C_47_hi);
    
    auto C_53_0 = my_select_repl_wit_t<less, ps, uncompr_f>::apply(replicated_lineorder.lo_quantity, 25, 28793053);
    auto C_53 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_53_0, C_47);
    
    auto X_55 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder.lo_orderdate, C_53);
    
    auto C_77 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date.d_year, 1993, 365);
    
    auto X_79 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date.d_datekey, C_77);
    
    auto X_81 = semi_join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >
    (X_79, X_55);
    
    auto X_89_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder.lo_extendedprice, C_53);
    X_89_0->template prepare_for_random_access<ps>();
    auto X_89 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_89_0, X_81);
    
    auto X_90_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder.lo_discount, C_53);
    X_90_0->template prepare_for_random_access<ps>();
    auto X_90 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_90_0, X_81);
    
    auto X_93 = morphstore::calc_binary<mul, ps, uncompr_f, uncompr_f, uncompr_f >(X_89, X_90);
    
    auto X_96 = agg_sum<ps, uncompr_f >(X_93);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
    delete replicated_lineorder_lo_discount_snapshot;
    delete C_47_lo;
    delete C_47_hi;
    delete C_47;
    delete replicated_lineorder_lo_quantity_snapshot;
    delete C_53_0;
    delete C_53;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_55;
    delete replicated_date_d_year_snapshot;
    delete C_77;
    delete replicated_date_d_datekey_snapshot;
    delete X_79;
    delete X_81;
    delete replicated_lineorder_lo_extendedprice_snapshot;
    delete X_89_0;
    delete X_89;
    delete X_90_0;
    delete X_90;
    delete X_93;
    delete X_96;

    return;
}

void query_q12(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_yearmonthnum_snapshot = ALUtils::take_snapshot(replicated_date.d_yearmonthnum);
    auto replicated_lineorder_lo_discount_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_discount);
    auto replicated_lineorder_lo_extendedprice_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_extendedprice);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_quantity_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_quantity);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_48_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_lineorder_lo_discount_snapshot, 4, 38175239);
    auto C_48_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_lineorder_lo_discount_snapshot, 6, 38174155);
    auto C_48 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_48_lo, C_48_hi);
    
    auto C_56_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_lineorder_lo_quantity_snapshot, 26, 29992885);
    auto C_56_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_lineorder_lo_quantity_snapshot, 35, 41992186);
    auto C_56_0 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_56_lo, C_56_hi);
    auto C_56 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_56_0, C_48);
    
    auto X_57 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, C_56);
    
    auto C_78 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_yearmonthnum_snapshot, 199401, 31);
    
    auto X_80 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_78);
    
    auto X_82 = semi_join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >
    (X_80, X_57);
    
    auto X_90_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_extendedprice_snapshot, C_56);
    X_90_0->template prepare_for_random_access<ps>();
    auto X_90 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_90_0, X_82);
    
    auto X_91_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_discount_snapshot, C_56);
    X_91_0->template prepare_for_random_access<ps>();
    auto X_91 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_91_0, X_82);
    
    auto X_94 = morphstore::calc_binary<mul, ps, uncompr_f, uncompr_f, uncompr_f >(X_90, X_91);
    
    auto X_97 = agg_sum<ps, uncompr_f >(X_94);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete C_48_lo;
    delete replicated_lineorder_lo_discount_snapshot;
    delete C_48_hi;
    delete C_48;
    delete C_56_lo;
    delete replicated_lineorder_lo_quantity_snapshot;
    delete C_56_hi;
    delete C_56_0;
    delete C_56;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_57;
    delete C_78;
    delete replicated_date_d_yearmonthnum_snapshot;
    delete replicated_date_d_datekey_snapshot;
    delete X_80;
    delete X_82;
    delete replicated_lineorder_lo_extendedprice_snapshot;
    delete X_90_0;
    delete X_90;
    delete X_91_0;
    delete X_91;
    delete X_94;
    delete X_97;
#endif

    return;
}

void query_q13(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_weeknuminyear_snapshot = ALUtils::take_snapshot(replicated_date.d_weeknuminyear);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_discount_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_discount);
    auto replicated_lineorder_lo_extendedprice_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_extendedprice);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_quantity_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_quantity);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_49_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_lineorder_lo_discount_snapshot, 5, 32719167);
    auto C_49_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_lineorder_lo_discount_snapshot, 7, 43627232);
    auto C_49 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_49_lo, C_49_hi);
    
    auto C_57_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_lineorder_lo_quantity_snapshot, 26, 29992885);
    auto C_57_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_lineorder_lo_quantity_snapshot, 35, 41992186);
    auto C_57_0 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_57_lo, C_57_hi);
    auto C_57 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_57_0, C_49);
    
    auto X_58 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, C_57);
    
    auto C_87 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_weeknuminyear_snapshot, 6, 49);
    
    auto C_91_0 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1994, 365);
    auto C_91 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_91_0, C_87);
    
    auto X_92 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_91);
    
    auto X_95 = semi_join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >
    (X_92, X_58);
    
    auto X_103_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_extendedprice_snapshot, C_57);
    X_103_0->template prepare_for_random_access<ps>();
    auto X_103 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_103_0, X_95);
    
    auto X_104_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_discount_snapshot, C_57);
    X_104_0->template prepare_for_random_access<ps>();
    auto X_104 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_104_0, X_95);
    
    auto X_108 = morphstore::calc_binary<mul, ps, uncompr_f, uncompr_f, uncompr_f >(X_103, X_104);
    
    auto X_111 = agg_sum<ps, uncompr_f >(X_108);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_lineorder_lo_discount_snapshot;
    delete C_49_lo;
    delete C_49_hi;
    delete C_49;
    delete replicated_lineorder_lo_quantity_snapshot;
    delete C_57_lo;
    delete C_57_hi;
    delete C_57_0;
    delete C_57;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_58;
    delete replicated_date_d_weeknuminyear_snapshot;
    delete C_87;
    delete replicated_date_d_year_snapshot;
    delete C_91_0;
    delete C_91;
    delete replicated_date_d_datekey_snapshot;
    delete X_92;
    delete X_95;
    delete replicated_lineorder_lo_extendedprice_snapshot;
    delete X_103_0;
    delete X_103;
    delete X_104_0;
    delete X_104;
    delete X_108;
    delete X_111;
#endif

    return;
}

void query_q21(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;

    // Take snapshots of all base data columns accessed by this query.
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_partkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_partkey);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_part_p_brand_snapshot = ALUtils::take_snapshot(replicated_part.p_brand);
    auto replicated_part_p_category_snapshot = ALUtils::take_snapshot(replicated_part.p_category);
    auto replicated_part_p_partkey_snapshot = ALUtils::take_snapshot(replicated_part.p_partkey);
    auto replicated_supplier_s_region_snapshot = ALUtils::take_snapshot(replicated_supplier.s_region);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_59 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_region_snapshot, 1, 4102);
    
    auto X_61 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_59);
    
    auto X_63 = semi_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(X_61, replicated_lineorder_lo_suppkey_snapshot);
    
    auto X_69 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_partkey_snapshot, X_63);
    
    auto C_100 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_category_snapshot, 1, 31882);
    
    auto X_102 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_partkey_snapshot, C_100);
    const column<uncompr_f > * X_106;
    const column<uncompr_f > * X_105;
    std::tie(X_106, X_105) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_102,
        X_69,
        X_69->get_count_values()
    );
    
    auto X_113_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, X_63);
    X_113_0->template prepare_for_random_access<ps>();
    auto X_113 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_113_0, X_105);
    
    const column<uncompr_f> * X_137;
    const column<uncompr_f> * X_136;
    std::tie(X_137, X_136) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        replicated_date_d_datekey_snapshot,
        X_113
    );
    

    auto X_146_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_brand_snapshot, C_100);
    X_146_0->template prepare_for_random_access<ps>();
    auto X_146_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_146_0, X_106);
    X_146_1->template prepare_for_random_access<ps>();
    auto X_146 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_146_1, X_136);
    X_146->template prepare_for_random_access<ps>();
    
    auto X_148 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, X_137);
    X_148->template prepare_for_random_access<ps>();
    const column<uncompr_f > * X_149;
    const column<uncompr_f > * C_150;
    std::tie(X_149, C_150) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_148);
    
    const column<uncompr_f > * X_152;
    const column<uncompr_f > * C_153;
    std::tie(X_152, C_153) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_149, X_146);

    auto X_155 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_148, C_153);
    
    auto X_156 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_146, C_153);
    
    auto X_141_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_63);
    X_141_0->template prepare_for_random_access<ps>();
    auto X_141_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_141_0, X_105);
    X_141_1->template prepare_for_random_access<ps>();
    auto X_141 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_141_1, X_136);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_157 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_152, X_141, C_153->get_count_values());
    


    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_supplier_s_region_snapshot;
    delete C_59;
    delete X_61;
    delete replicated_supplier_s_suppkey_snapshot;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_63;
    delete X_69;
    delete replicated_lineorder_lo_partkey_snapshot;
    delete replicated_part_p_category_snapshot;
    delete C_100;
    delete X_102;
    delete replicated_part_p_partkey_snapshot;
    delete X_106;
    delete X_105;
    delete X_113_0;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_113;
    delete X_137;
    delete replicated_date_d_datekey_snapshot;
    delete X_136;
    delete X_146_0;
    delete replicated_part_p_brand_snapshot;
    delete X_146_1;
    delete X_146;
    delete X_148;
    delete replicated_date_d_year_snapshot;
    delete C_150;
    delete X_149;
    delete X_152;
    delete C_153;
    delete X_155;
    delete X_156;
    delete X_141_0;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_141_1;
    delete X_141;
    delete X_157;
#endif

    return;
}

void query_q22(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_partkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_partkey);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_part_p_brand_snapshot = ALUtils::take_snapshot(replicated_part.p_brand);
    auto replicated_part_p_partkey_snapshot = ALUtils::take_snapshot(replicated_part.p_partkey);
    auto replicated_supplier_s_region_snapshot = ALUtils::take_snapshot(replicated_supplier.s_region);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_60 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_region_snapshot, 2, 4001);
    
    auto X_62 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_60);
    
    auto X_64 = semi_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(X_62, replicated_lineorder_lo_suppkey_snapshot);
    
    auto X_70 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_partkey_snapshot, X_64);
    
    auto C_96_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_part_p_brand_snapshot, 253, 597886);
    auto C_96_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_part_p_brand_snapshot, 260, 208523);
    auto C_96 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_96_lo, C_96_hi);
    
    auto X_99 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_partkey_snapshot, C_96);
    
    const column<uncompr_f > * X_102;
    const column<uncompr_f > * X_101;
    std::tie(X_102, X_101) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_99,
        X_70,
        X_70->get_count_values()
    );
    
    
    auto X_109_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, X_64);
    X_109_0->template prepare_for_random_access<ps>();
    auto X_109 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_109_0, X_101);
    
    const column<uncompr_f> * X_132;
    const column<uncompr_f> * X_131;
    std::tie(X_132, X_131) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        replicated_date_d_datekey_snapshot,
        X_109
    );
    
    
    auto X_140_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_brand_snapshot, C_96);
    X_140_0->template prepare_for_random_access<ps>();
    auto X_140_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_140_0, X_102);
    X_140_1->template prepare_for_random_access<ps>();
    auto X_140 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_140_1, X_131);
    X_140->template prepare_for_random_access<ps>();
    
    auto X_142 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, X_132);
    X_142->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_143;
    const column<uncompr_f > * C_144;
    std::tie(X_143, C_144) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_142);
    
    const column<uncompr_f > * X_146;
    const column<uncompr_f > * C_147;
    std::tie(X_146, C_147) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_143, X_140);
    
    auto X_149 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_142, C_147);
    
    auto X_150 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_140, C_147);
    
    auto X_136_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_64);
    X_136_0->template prepare_for_random_access<ps>();
    auto X_136_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_136_0, X_101);
    X_136_1->template prepare_for_random_access<ps>();
    auto X_136 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_136_1, X_131);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_151 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_146, X_136, C_147->get_count_values());
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete C_60;
    delete replicated_supplier_s_region_snapshot;
    delete X_62;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_64;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_70;
    delete replicated_lineorder_lo_partkey_snapshot;
    delete C_96_lo;
    delete replicated_part_p_brand_snapshot;
    delete C_96_hi;
    delete C_96;
    delete X_99;
    delete replicated_part_p_partkey_snapshot;
    delete X_101;
    delete X_102;
    delete X_109_0;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_109;
    delete replicated_date_d_datekey_snapshot;
    delete X_131;
    delete X_132;
    delete X_140_0;
    delete X_140_1;
    delete X_140;
    delete X_142;
    delete replicated_date_d_year_snapshot;
    delete X_143;
    delete C_144;
    delete X_146;
    delete C_147;
    delete X_149;
    delete X_150;
    delete X_136_0;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_136_1;
    delete X_136;
    delete X_151;
#endif

    return;
}

void query_q23(ThreadData* td)
{
   using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_partkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_partkey);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_part_p_brand_snapshot = ALUtils::take_snapshot(replicated_part.p_brand);
    auto replicated_part_p_partkey_snapshot = ALUtils::take_snapshot(replicated_part.p_partkey);
    auto replicated_supplier_s_region_snapshot = ALUtils::take_snapshot(replicated_supplier.s_region);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_59 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_region_snapshot, 3, 3972);
    
    auto X_61 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_59);
    
    auto X_63 = semi_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(X_61, replicated_lineorder_lo_suppkey_snapshot);
    
    auto X_69 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_partkey_snapshot, X_63);
    
    auto C_93 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_brand_snapshot, 272, 794);
    
    auto X_95 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_partkey_snapshot, C_93);
    
    const column<uncompr_f > * X_98;
    const column<uncompr_f > * X_97;
    std::tie(X_98, X_97) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_95,
        X_69,
        X_69->get_count_values()
    );
    
    
    auto X_105_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, X_63);
    X_105_0->template prepare_for_random_access<ps>();
    auto X_105 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_105_0, X_97);
    
    const column<uncompr_f> * X_128;
    const column<uncompr_f> * X_127;
    std::tie(X_128, X_127) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        replicated_date_d_datekey_snapshot,
        X_105
    );
    
    
    auto X_136_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_brand_snapshot, C_93);
    X_136_0->template prepare_for_random_access<ps>();
    auto X_136_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_136_0, X_98);
    X_136_1->template prepare_for_random_access<ps>();
    auto X_136 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_136_1, X_127);
    X_136->template prepare_for_random_access<ps>();
    
    auto X_138 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, X_128);
    X_138->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_139;
    const column<uncompr_f > * C_140;
    std::tie(X_139, C_140) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_138);
    
    const column<uncompr_f > * X_142;
    const column<uncompr_f > * C_143;
    std::tie(X_142, C_143) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_139, X_136);
    
    auto X_145 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_138, C_143);
    
    auto X_146 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_136, C_143);
    
    auto X_132_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_63);
    X_132_0->template prepare_for_random_access<ps>();
    auto X_132_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_132_0, X_97);
    X_132_1->template prepare_for_random_access<ps>();
    auto X_132 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_132_1, X_127);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_147 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_142, X_132, C_143->get_count_values());
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete C_59;
    delete replicated_supplier_s_region_snapshot;
    delete X_61;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_63;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_69;
    delete replicated_lineorder_lo_partkey_snapshot;
    delete C_93;
    delete replicated_part_p_brand_snapshot;
    delete X_95;
    delete replicated_part_p_partkey_snapshot;
    delete X_97;
    delete X_98;
    delete X_105_0;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_105;
    delete X_127;
    delete replicated_date_d_datekey_snapshot;
    delete X_128;
    delete X_136_0;
    delete X_136_1;
    delete X_136;
    delete X_138;
    delete replicated_date_d_year_snapshot;
    delete C_140;
    delete X_139;
    delete X_142;
    delete C_143;
    delete X_145;
    delete X_146;
    delete X_132_0;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_132_1;
    delete X_132;
    delete X_147;
#endif

    return;
}

void query_q31(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_customer_c_nation_snapshot = ALUtils::take_snapshot(replicated_customer.c_nation);
    auto replicated_customer_c_region_snapshot = ALUtils::take_snapshot(replicated_customer.c_region);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_supplier_s_nation_snapshot = ALUtils::take_snapshot(replicated_supplier.s_nation);
    auto replicated_supplier_s_region_snapshot = ALUtils::take_snapshot(replicated_supplier.s_region);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_63_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1992, 2556);
    auto C_63_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1997, 2192);
    auto C_63 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_63_lo, C_63_hi);
    
    auto X_67 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_63);
    
    const column<uncompr_f> * X_70;
    const column<uncompr_f> * X_69;
    std::tie(X_70, X_69) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        X_67,
        replicated_lineorder_lo_orderdate_snapshot
    );
    
    
    auto X_75 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_69);
    
    auto C_105 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_region_snapshot, 2, 4001);
    
    auto X_107 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_105);
    
    const column<uncompr_f > * X_111;
    const column<uncompr_f > * X_110;
    std::tie(X_111, X_110) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_107,
        X_75,
        X_75->get_count_values()
    );
    
    
    auto X_116_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_custkey_snapshot, X_69);
    X_116_0->template prepare_for_random_access<ps>();
    auto X_116 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_116_0, X_110);
    
    auto C_150 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_region_snapshot, 2, 60535);
    
    auto X_152 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_150);
    
    const column<uncompr_f > * X_156;
    const column<uncompr_f > * X_155;
    std::tie(X_156, X_155) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_152,
        X_116,
        X_116->get_count_values()
    );
    
    
    auto X_164_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_69);
    X_164_0->template prepare_for_random_access<ps>();
    auto X_164_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_164_0, X_110);
    X_164_1->template prepare_for_random_access<ps>();
    auto X_164 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_164_1, X_155);
    
    auto X_166_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, C_63);
    X_166_0->template prepare_for_random_access<ps>();
    auto X_166_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166_0, X_70);
    X_166_1->template prepare_for_random_access<ps>();
    auto X_166_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166_1, X_110);
    X_166_2->template prepare_for_random_access<ps>();
    auto X_166 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166_2, X_155);
    X_166->template prepare_for_random_access<ps>();
    
    auto X_168_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_nation_snapshot, C_105);
    X_168_0->template prepare_for_random_access<ps>();
    auto X_168_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_168_0, X_111);
    X_168_1->template prepare_for_random_access<ps>();
    auto X_168 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_168_1, X_155);
    X_168->template prepare_for_random_access<ps>();
    
    auto X_171_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_nation_snapshot, C_150);
    X_171_0->template prepare_for_random_access<ps>();
    auto X_171 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_171_0, X_156);
    X_171->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_173;
    const column<uncompr_f > * C_174;
    std::tie(X_173, C_174) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_171);
    
    const column<uncompr_f > * X_176;
    const column<uncompr_f > * C_177;
    std::tie(X_176, C_177) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_173, X_168);
    
    const column<uncompr_f > * X_179;
    const column<uncompr_f > * C_180;
    std::tie(X_179, C_180) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_176, X_166);
    
    auto X_184 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166, C_180);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_185 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_179, X_164, C_180->get_count_values());
    
    auto X_195 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_171, C_180);
    
    auto X_196 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_168, C_180);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_date_d_year_snapshot;
    delete C_63_lo;
    delete C_63_hi;
    delete C_63;
    delete X_67;
    delete replicated_date_d_datekey_snapshot;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_70;
    delete X_69;
    delete X_75;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete replicated_supplier_s_region_snapshot;
    delete C_105;
    delete X_107;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_111;
    delete X_110;
    delete X_116_0;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete X_116;
    delete replicated_customer_c_region_snapshot;
    delete C_150;
    delete X_152;
    delete replicated_customer_c_custkey_snapshot;
    delete X_156;
    delete X_155;
    delete X_164_0;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_164_1;
    delete X_164;
    delete X_166_0;
    delete X_166_1;
    delete X_166_2;
    delete X_166;
    delete X_168_0;
    delete replicated_supplier_s_nation_snapshot;
    delete X_168_1;
    delete X_168;
    delete X_171_0;
    delete replicated_customer_c_nation_snapshot;
    delete X_171;
    delete X_173;
    delete C_174;
    delete X_176;
    delete C_177;
    delete X_179;
    delete C_180;
    delete X_184;
    delete X_185;
    delete X_195;
    delete X_196;
#endif

    return;
}

void query_q32(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_city_snapshot = ALUtils::take_snapshot(replicated_customer.c_city);
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_customer_c_nation_snapshot = ALUtils::take_snapshot(replicated_customer.c_nation);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_supplier_s_city_snapshot = ALUtils::take_snapshot(replicated_supplier.s_city);
    auto replicated_supplier_s_nation_snapshot = ALUtils::take_snapshot(replicated_supplier.s_nation);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_63_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1992, 2556);
    auto C_63_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1997, 2192);
    auto C_63 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_63_lo, C_63_hi);
    
    auto X_67 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_63);
    
    const column<uncompr_f> * X_70;
    const column<uncompr_f> * X_69;
    std::tie(X_70, X_69) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        X_67,
        replicated_lineorder_lo_orderdate_snapshot
    );
    
    
    auto X_75 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_69);
    
    auto C_105 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_nation_snapshot, 23, 809);
    
    auto X_107 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_105);
    
    const column<uncompr_f > * X_111;
    const column<uncompr_f > * X_110;
    std::tie(X_111, X_110) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_107,
        X_75,
        X_75->get_count_values()
    );
    
    
    auto X_116_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_custkey_snapshot, X_69);
    X_116_0->template prepare_for_random_access<ps>();
    auto X_116 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_116_0, X_110);
    
    auto C_150 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_nation_snapshot, 23, 11913);
    
    auto X_152 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_150);
    
    const column<uncompr_f > * X_156;
    const column<uncompr_f > * X_155;
    std::tie(X_156, X_155) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_152,
        X_116,
        X_116->get_count_values()
    );
    
    
    auto X_164_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_69);
    X_164_0->template prepare_for_random_access<ps>();
    auto X_164_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_164_0, X_110);
    X_164_1->template prepare_for_random_access<ps>();
    auto X_164 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_164_1, X_155);
    
    auto X_166_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, C_63);
    X_166_0->template prepare_for_random_access<ps>();
    auto X_166_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166_0, X_70);
    X_166_1->template prepare_for_random_access<ps>();
    auto X_166_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166_1, X_110);
    X_166_2->template prepare_for_random_access<ps>();
    auto X_166 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166_2, X_155);
    X_166->template prepare_for_random_access<ps>();
    
    auto X_168_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_city_snapshot, C_105);
    X_168_0->template prepare_for_random_access<ps>();
    auto X_168_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_168_0, X_111);
    X_168_1->template prepare_for_random_access<ps>();
    auto X_168 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_168_1, X_155);
    X_168->template prepare_for_random_access<ps>();
    
    auto X_171_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_city_snapshot, C_150);
    X_171_0->template prepare_for_random_access<ps>();
    auto X_171 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_171_0, X_156);
    X_171->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_173;
    const column<uncompr_f > * C_174;
    std::tie(X_173, C_174) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_171);
    
    const column<uncompr_f > * X_176;
    const column<uncompr_f > * C_177;
    std::tie(X_176, C_177) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_173, X_168);
    
    const column<uncompr_f > * X_179;
    const column<uncompr_f > * C_180;
    std::tie(X_179, C_180) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_176, X_166);
    
    auto X_184 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_166, C_180);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_185 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_179, X_164, C_180->get_count_values());
    
    auto X_195 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_171, C_180);
    
    auto X_196 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_168, C_180);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_date_d_year_snapshot;
    delete C_63_lo;
    delete C_63_hi;
    delete C_63;
    delete replicated_date_d_datekey_snapshot;
    delete X_67;
    delete X_70;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_69;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_75;
    delete replicated_supplier_s_nation_snapshot;
    delete C_105;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_107;
    delete X_111;
    delete X_110;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete X_116_0;
    delete X_116;
    delete replicated_customer_c_nation_snapshot;
    delete C_150;
    delete replicated_customer_c_custkey_snapshot;
    delete X_152;
    delete X_156;
    delete X_155;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_164_0;
    delete X_164_1;
    delete X_164;
    delete X_166_0;
    delete X_166_1;
    delete X_166_2;
    delete X_166;
    delete replicated_supplier_s_city_snapshot;
    delete X_168_0;
    delete X_168_1;
    delete X_168;
    delete replicated_customer_c_city_snapshot;
    delete X_171_0;
    delete X_171;
    delete C_174;
    delete X_173;
    delete C_177;
    delete X_176;
    delete C_180;
    delete X_179;
    delete X_184;
    delete X_185;
    delete X_195;
    delete X_196;
#endif

    return;
}

void query_q33(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_city_snapshot = ALUtils::take_snapshot(replicated_customer.c_city);
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_supplier_s_city_snapshot = ALUtils::take_snapshot(replicated_supplier.s_city);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_63 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_city_snapshot, 221, 1183);
    
    auto C_67 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_city_snapshot, 225, 1187);
    
    auto C_68 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_63, C_67);
    
    auto X_69 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_68);
    
    const column<uncompr_f> * X_72;
    const column<uncompr_f> * X_71;
    std::tie(X_72, X_71) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        X_69,
        replicated_lineorder_lo_custkey_snapshot
    );
    
    
    auto X_78 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_71);
    
    auto C_101 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_city_snapshot, 221, 77);
    
    auto C_105 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_city_snapshot, 225, 87);
    
    auto C_106 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_101, C_105);
    
    auto X_107 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_106);
    
    const column<uncompr_f > * X_110;
    const column<uncompr_f > * X_109;
    std::tie(X_110, X_109) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_107,
        X_78,
        X_78->get_count_values()
    );
    
    
    auto X_117_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, X_71);
    X_117_0->template prepare_for_random_access<ps>();
    auto X_117 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_117_0, X_109);
    
    auto C_143_lo = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1992, 2556);
    auto C_143_hi = my_select_repl_wit_t<lessequal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1997, 2192);
    auto C_143 = intersect_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_143_lo, C_143_hi);
    
    auto X_147 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_143);
    
    const column<uncompr_f > * X_150;
    const column<uncompr_f > * X_149;
    std::tie(X_150, X_149) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_147,
        X_117,
        X_117->get_count_values()
    );
    
    
    auto X_157_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_71);
    X_157_0->template prepare_for_random_access<ps>();
    auto X_157_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_157_0, X_109);
    X_157_1->template prepare_for_random_access<ps>();
    auto X_157 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_157_1, X_149);
    
    auto X_163_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, C_143);
    X_163_0->template prepare_for_random_access<ps>();
    auto X_163 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_163_0, X_150);
    X_163->template prepare_for_random_access<ps>();
    
    auto X_161_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_city_snapshot, C_106);
    X_161_0->template prepare_for_random_access<ps>();
    auto X_161_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_161_0, X_110);
    X_161_1->template prepare_for_random_access<ps>();
    auto X_161 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_161_1, X_149);
    X_161->template prepare_for_random_access<ps>();
    
    auto X_159_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_city_snapshot, C_68);
    X_159_0->template prepare_for_random_access<ps>();
    auto X_159_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_159_0, X_72);
    X_159_1->template prepare_for_random_access<ps>();
    auto X_159_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_159_1, X_109);
    X_159_2->template prepare_for_random_access<ps>();
    auto X_159 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_159_2, X_149);
    X_159->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_164;
    const column<uncompr_f > * C_165;
    std::tie(X_164, C_165) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_159);
    
    const column<uncompr_f > * X_167;
    const column<uncompr_f > * C_168;
    std::tie(X_167, C_168) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_164, X_161);
    
    const column<uncompr_f > * X_170;
    const column<uncompr_f > * C_171;
    std::tie(X_170, C_171) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_167, X_163);
    
    auto X_175 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_163, C_171);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_176 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_170, X_157, C_171->get_count_values());
    
    auto X_186 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_159, C_171);
    
    auto X_187 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_161, C_171);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_customer_c_city_snapshot;
    delete C_63;
    delete C_67;
    delete C_68;
    delete replicated_customer_c_custkey_snapshot;
    delete X_69;
    delete X_72;
    delete X_71;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_78;
    delete replicated_supplier_s_city_snapshot;
    delete C_101;
    delete C_105;
    delete C_106;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_107;
    delete X_110;
    delete X_109;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_117_0;
    delete X_117;
    delete replicated_date_d_year_snapshot;
    delete C_143_lo;
    delete C_143_hi;
    delete C_143;
    delete replicated_date_d_datekey_snapshot;
    delete X_147;
    delete X_150;
    delete X_149;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_157_0;
    delete X_157_1;
    delete X_157;
    delete X_163_0;
    delete X_163;
    delete X_161_0;
    delete X_161_1;
    delete X_161;
    delete X_159_0;
    delete X_159_1;
    delete X_159_2;
    delete X_159;
    delete X_164;
    delete C_165;
    delete X_167;
    delete C_168;
    delete X_170;
    delete C_171;
    delete X_175;
    delete X_176;
    delete X_186;
    delete X_187;
#endif

    return;
}

void query_q34(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_city_snapshot = ALUtils::take_snapshot(replicated_customer.c_city);
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_date_d_yearmonth_snapshot = ALUtils::take_snapshot(replicated_date.d_yearmonth);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_supplier_s_city_snapshot = ALUtils::take_snapshot(replicated_supplier.s_city);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_62 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_city_snapshot, 221, 1183);
    
    auto C_66 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_city_snapshot, 225, 1187);
    
    auto C_67 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_62, C_66);
    
    auto X_68 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_67);
    
    const column<uncompr_f> * X_71;
    const column<uncompr_f> * X_70;
    std::tie(X_71, X_70) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        X_68,
        replicated_lineorder_lo_custkey_snapshot
    );
    
    
    auto X_77 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_70);
    
    auto C_100 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_city_snapshot, 221, 77);
    
    auto C_104 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_city_snapshot, 225, 87);
    
    auto C_105 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_100, C_104);
    
    auto X_106 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_105);
    
    const column<uncompr_f > * X_109;
    const column<uncompr_f > * X_108;
    std::tie(X_109, X_108) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_106,
        X_77,
        X_77->get_count_values()
    );
    
    
    auto X_116_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, X_70);
    X_116_0->template prepare_for_random_access<ps>();
    auto X_116 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_116_0, X_108);
    
    auto C_147 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_yearmonth_snapshot, 19, 31);
    
    auto X_149 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_147);
    
    const column<uncompr_f > * X_153;
    const column<uncompr_f > * X_152;
    std::tie(X_153, X_152) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_149,
        X_116,
        X_116->get_count_values()
    );
    
    
    auto X_161_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_70);
    X_161_0->template prepare_for_random_access<ps>();
    auto X_161_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_161_0, X_108);
    X_161_1->template prepare_for_random_access<ps>();
    auto X_161 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_161_1, X_152);
    
    auto X_165_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_city_snapshot, C_105);
    X_165_0->template prepare_for_random_access<ps>();
    auto X_165_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_165_0, X_109);
    X_165_1->template prepare_for_random_access<ps>();
    auto X_165 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_165_1, X_152);
    X_165->template prepare_for_random_access<ps>();
    
    auto X_163_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_city_snapshot, C_67);
    X_163_0->template prepare_for_random_access<ps>();
    auto X_163_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_163_0, X_71);
    X_163_1->template prepare_for_random_access<ps>();
    auto X_163_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_163_1, X_108);
    X_163_2->template prepare_for_random_access<ps>();
    auto X_163 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_163_2, X_152);
    X_163->template prepare_for_random_access<ps>();
    
    auto X_167_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, C_147);
    X_167_0->template prepare_for_random_access<ps>();
    auto X_167 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_167_0, X_153);
    X_167->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_169;
    const column<uncompr_f > * C_170;
    std::tie(X_169, C_170) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_167);
    
    const column<uncompr_f > * X_172;
    const column<uncompr_f > * C_173;
    std::tie(X_172, C_173) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_169, X_163);
    
    const column<uncompr_f > * X_175;
    const column<uncompr_f > * C_176;
    std::tie(X_175, C_176) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_172, X_165);
    
    auto X_180 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_167, C_176);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_181 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_175, X_161, C_176->get_count_values());
    
    auto X_191 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_163, C_176);
    
    auto X_192 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_165, C_176);
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_customer_c_city_snapshot;
    delete C_62;
    delete C_66;
    delete C_67;
    delete replicated_customer_c_custkey_snapshot;
    delete X_68;
    delete X_70;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete X_71;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_77;
    delete replicated_supplier_s_city_snapshot;
    delete C_100;
    delete C_104;
    delete C_105;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_106;
    delete X_108;
    delete X_109;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_116_0;
    delete X_116;
    delete replicated_date_d_yearmonth_snapshot;
    delete C_147;
    delete replicated_date_d_datekey_snapshot;
    delete X_149;
    delete X_152;
    delete X_153;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_161_0;
    delete X_161_1;
    delete X_161;
    delete X_165_0;
    delete X_165_1;
    delete X_165;
    delete X_163_0;
    delete X_163_1;
    delete X_163_2;
    delete X_163;
    delete replicated_date_d_year_snapshot;
    delete X_167_0;
    delete X_167;
    delete C_170;
    delete X_169;
    delete C_173;
    delete X_172;
    delete C_176;
    delete X_175;
    delete X_180;
    delete X_181;
    delete X_191;
    delete X_192;
#endif

    return;
}

void query_q41(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_customer_c_nation_snapshot = ALUtils::take_snapshot(replicated_customer.c_nation);
    auto replicated_customer_c_region_snapshot = ALUtils::take_snapshot(replicated_customer.c_region);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_partkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_partkey);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_lineorder_lo_supplycost_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_supplycost);
    auto replicated_part_p_mfgr_snapshot = ALUtils::take_snapshot(replicated_part.p_mfgr);
    auto replicated_part_p_partkey_snapshot = ALUtils::take_snapshot(replicated_part.p_partkey);
    auto replicated_supplier_s_region_snapshot = ALUtils::take_snapshot(replicated_supplier.s_region);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_75 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_mfgr_snapshot, 0, 160027);
    
    auto C_79 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_mfgr_snapshot, 1, 159744);
    
    auto C_80 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_75, C_79);
    
    auto X_81 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_partkey_snapshot, C_80);
    
    auto X_83 = semi_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(X_81, replicated_lineorder_lo_partkey_snapshot);
    
    auto X_91 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_83);
    
    auto C_115 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_region_snapshot, 1, 4102);
    
    auto X_117 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_115);
    
    auto X_119 = semi_join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >
    (X_117, X_91);
    
    auto X_125_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_custkey_snapshot, X_83);
    X_125_0->template prepare_for_random_access<ps>();
    auto X_125 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_125_0, X_119);
    
    auto C_160 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_region_snapshot, 1, 59761);
    
    auto X_162 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_160);
    
    const column<uncompr_f > * X_166;
    const column<uncompr_f > * X_165;
    std::tie(X_166, X_165) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_162,
        X_125,
        X_125->get_count_values()
    );
    
    
    auto X_174_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_orderdate_snapshot, X_83);
    X_174_0->template prepare_for_random_access<ps>();
    auto X_174_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_174_0, X_119);
    X_174_1->template prepare_for_random_access<ps>();
    auto X_174 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_174_1, X_165);
    
    const column<uncompr_f> * X_201;
    const column<uncompr_f> * X_200;
    std::tie(X_201, X_200) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        replicated_date_d_datekey_snapshot,
        X_174
    );
    
    
    auto X_217_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_nation_snapshot, C_160);
    X_217_0->template prepare_for_random_access<ps>();
    auto X_217_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_217_0, X_166);
    X_217_1->template prepare_for_random_access<ps>();
    auto X_217 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_217_1, X_200);
    X_217->template prepare_for_random_access<ps>();
    
    auto X_220 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, X_201);
    X_220->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_226;
    const column<uncompr_f > * C_227;
    std::tie(X_226, C_227) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_220);
    
    const column<uncompr_f > * X_229;
    const column<uncompr_f > * C_230;
    std::tie(X_229, C_230) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_226, X_217);
    
    auto X_232 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_220, C_230);
    
    auto X_233 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_217, C_230);
    
    auto X_210_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_83);
    X_210_0->template prepare_for_random_access<ps>();
    auto X_210_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_210_0, X_119);
    X_210_1->template prepare_for_random_access<ps>();
    auto X_210_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_210_1, X_165);
    X_210_2->template prepare_for_random_access<ps>();
    auto X_210 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_210_2, X_200);
    
    auto X_211_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_supplycost_snapshot, X_83);
    X_211_0->template prepare_for_random_access<ps>();
    auto X_211_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_211_0, X_119);
    X_211_1->template prepare_for_random_access<ps>();
    auto X_211_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_211_1, X_165);
    X_211_2->template prepare_for_random_access<ps>();
    auto X_211 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_211_2, X_200);
    
    auto X_223 = morphstore::calc_binary<sub, ps, uncompr_f, uncompr_f, uncompr_f >(X_210, X_211);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_234 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_229, X_223, C_230->get_count_values());
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete replicated_part_p_mfgr_snapshot;
    delete C_75;
    delete C_79;
    delete C_80;
    delete replicated_part_p_partkey_snapshot;
    delete X_81;
    delete X_83;
    delete replicated_lineorder_lo_partkey_snapshot;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_91;
    delete replicated_supplier_s_region_snapshot;
    delete C_115;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_117;
    delete X_119;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete X_125_0;
    delete X_125;
    delete replicated_customer_c_region_snapshot;
    delete C_160;
    delete replicated_customer_c_custkey_snapshot;
    delete X_162;
    delete X_165;
    delete X_166;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_174_0;
    delete X_174_1;
    delete X_174;
    delete X_200;
    delete replicated_date_d_datekey_snapshot;
    delete X_201;
    delete replicated_customer_c_nation_snapshot;
    delete X_217_0;
    delete X_217_1;
    delete X_217;
    delete replicated_date_d_year_snapshot;
    delete X_220;
    delete C_227;
    delete X_226;
    delete C_230;
    delete X_229;
    delete X_232;
    delete X_233;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_210_0;
    delete X_210_1;
    delete X_210_2;
    delete X_210;
    delete replicated_lineorder_lo_supplycost_snapshot;
    delete X_211_0;
    delete X_211_1;
    delete X_211_2;
    delete X_211;
    delete X_223;
    delete X_234;
#endif

    return;
}

void query_q42(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_customer_c_region_snapshot = ALUtils::take_snapshot(replicated_customer.c_region);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_partkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_partkey);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_lineorder_lo_supplycost_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_supplycost);
    auto replicated_part_p_category_snapshot = ALUtils::take_snapshot(replicated_part.p_category);
    auto replicated_part_p_mfgr_snapshot = ALUtils::take_snapshot(replicated_part.p_mfgr);
    auto replicated_part_p_partkey_snapshot = ALUtils::take_snapshot(replicated_part.p_partkey);
    auto replicated_supplier_s_nation_snapshot = ALUtils::take_snapshot(replicated_supplier.s_nation);
    auto replicated_supplier_s_region_snapshot = ALUtils::take_snapshot(replicated_supplier.s_region);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_77 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1997, 365);
    
    auto C_81 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1998, 364);
    
    auto C_82 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_77, C_81);
    
    auto X_83 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_82);
    
    const column<uncompr_f> * X_86;
    const column<uncompr_f> * X_85;
    std::tie(X_86, X_85) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        X_83,
        replicated_lineorder_lo_orderdate_snapshot
    );
    
    
    auto X_92 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_partkey_snapshot, X_85);
    
    auto C_124 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_mfgr_snapshot, 0, 160027);
    
    auto C_128 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_mfgr_snapshot, 1, 159744);
    
    auto C_129 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_124, C_128);
    
    auto X_130 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_partkey_snapshot, C_129);
    
    const column<uncompr_f > * X_134;
    const column<uncompr_f > * X_133;
    std::tie(X_134, X_133) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_130,
        X_92,
        X_92->get_count_values()
    );
    
    
    auto X_141_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_85);
    X_141_0->template prepare_for_random_access<ps>();
    auto X_141 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_141_0, X_133);
    
    auto C_175 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_region_snapshot, 1, 4102);
    
    auto X_177 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_175);
    
    const column<uncompr_f > * X_181;
    const column<uncompr_f > * X_180;
    std::tie(X_181, X_180) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_177,
        X_141,
        X_141->get_count_values()
    );
    
    
    auto X_186_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_custkey_snapshot, X_85);
    X_186_0->template prepare_for_random_access<ps>();
    auto X_186_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_186_0, X_133);
    X_186_1->template prepare_for_random_access<ps>();
    auto X_186 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_186_1, X_180);
    
    auto C_218 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_region_snapshot, 1, 59761);
    
    auto X_220 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_218);
    
    auto X_222 = semi_join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >
    (X_220, X_186);
    
    auto X_238_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_category_snapshot, C_129);
    X_238_0->template prepare_for_random_access<ps>();
    auto X_238_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_238_0, X_134);
    X_238_1->template prepare_for_random_access<ps>();
    auto X_238_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_238_1, X_180);
    X_238_2->template prepare_for_random_access<ps>();
    auto X_238 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_238_2, X_222);
    X_238->template prepare_for_random_access<ps>();
    
    auto X_240_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_nation_snapshot, C_175);
    X_240_0->template prepare_for_random_access<ps>();
    auto X_240_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_240_0, X_181);
    X_240_1->template prepare_for_random_access<ps>();
    auto X_240 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_240_1, X_222);
    X_240->template prepare_for_random_access<ps>();
    
    auto X_235_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, C_82);
    X_235_0->template prepare_for_random_access<ps>();
    auto X_235_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235_0, X_86);
    X_235_1->template prepare_for_random_access<ps>();
    auto X_235_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235_1, X_133);
    X_235_2->template prepare_for_random_access<ps>();
    auto X_235_3 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235_2, X_180);
    X_235_3->template prepare_for_random_access<ps>();
    auto X_235 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235_3, X_222);
    X_235->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_249;
    const column<uncompr_f > * C_250;
    std::tie(X_249, C_250) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_235);
    
    const column<uncompr_f > * X_252;
    const column<uncompr_f > * C_253;
    std::tie(X_252, C_253) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_249, X_240);
    
    const column<uncompr_f > * X_255;
    const column<uncompr_f > * C_256;
    std::tie(X_255, C_256) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_252, X_238);
    
    auto X_258 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235, C_256);
    
    auto X_259 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_240, C_256);
    
    auto X_260 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_238, C_256);
    
    auto X_232_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_85);
    X_232_0->template prepare_for_random_access<ps>();
    auto X_232_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_232_0, X_133);
    X_232_1->template prepare_for_random_access<ps>();
    auto X_232_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_232_1, X_180);
    X_232_2->template prepare_for_random_access<ps>();
    auto X_232 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_232_2, X_222);
    
    auto X_233_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_supplycost_snapshot, X_85);
    X_233_0->template prepare_for_random_access<ps>();
    auto X_233_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233_0, X_133);
    X_233_1->template prepare_for_random_access<ps>();
    auto X_233_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233_1, X_180);
    X_233_2->template prepare_for_random_access<ps>();
    auto X_233 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233_2, X_222);
    
    auto X_246 = morphstore::calc_binary<sub, ps, uncompr_f, uncompr_f, uncompr_f >(X_232, X_233);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_261 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_255, X_246, C_256->get_count_values());
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete C_77;
    delete replicated_date_d_year_snapshot;
    delete C_81;
    delete C_82;
    delete X_83;
    delete replicated_date_d_datekey_snapshot;
    delete X_85;
    delete X_86;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_92;
    delete replicated_lineorder_lo_partkey_snapshot;
    delete C_124;
    delete replicated_part_p_mfgr_snapshot;
    delete C_128;
    delete C_129;
    delete X_130;
    delete replicated_part_p_partkey_snapshot;
    delete X_133;
    delete X_134;
    delete X_141_0;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_141;
    delete C_175;
    delete replicated_supplier_s_region_snapshot;
    delete X_177;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_180;
    delete X_181;
    delete X_186_0;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete X_186_1;
    delete X_186;
    delete C_218;
    delete replicated_customer_c_region_snapshot;
    delete X_220;
    delete replicated_customer_c_custkey_snapshot;
    delete X_222;
    delete X_238_0;
    delete replicated_part_p_category_snapshot;
    delete X_238_1;
    delete X_238_2;
    delete X_238;
    delete X_240_0;
    delete replicated_supplier_s_nation_snapshot;
    delete X_240_1;
    delete X_240;
    delete X_235_0;
    delete X_235_1;
    delete X_235_2;
    delete X_235_3;
    delete X_235;
    delete X_249;
    delete C_250;
    delete X_252;
    delete C_253;
    delete X_255;
    delete C_256;
    delete X_258;
    delete X_259;
    delete X_260;
    delete X_232_0;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_232_1;
    delete X_232_2;
    delete X_232;
    delete X_233_0;
    delete replicated_lineorder_lo_supplycost_snapshot;
    delete X_233_1;
    delete X_233_2;
    delete X_233;
    delete X_246;
    delete X_261;
#endif

    return;
}

void query_q43(ThreadData* td)
{
    using ps = scalar<v64<uint64_t>>;


    // Take snapshots of all base data columns accessed by this query.
    auto replicated_customer_c_custkey_snapshot = ALUtils::take_snapshot(replicated_customer.c_custkey);
    auto replicated_customer_c_region_snapshot = ALUtils::take_snapshot(replicated_customer.c_region);
    auto replicated_date_d_datekey_snapshot = ALUtils::take_snapshot(replicated_date.d_datekey);
    auto replicated_date_d_year_snapshot = ALUtils::take_snapshot(replicated_date.d_year);
    auto replicated_lineorder_lo_custkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_custkey);
    auto replicated_lineorder_lo_orderdate_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_orderdate);
    auto replicated_lineorder_lo_partkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_partkey);
    auto replicated_lineorder_lo_revenue_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_revenue);
    auto replicated_lineorder_lo_suppkey_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_suppkey);
    auto replicated_lineorder_lo_supplycost_snapshot = ALUtils::take_snapshot(replicated_lineorder.lo_supplycost);
    auto replicated_part_p_brand_snapshot = ALUtils::take_snapshot(replicated_part.p_brand);
    auto replicated_part_p_category_snapshot = ALUtils::take_snapshot(replicated_part.p_category);
    auto replicated_part_p_partkey_snapshot = ALUtils::take_snapshot(replicated_part.p_partkey);
    auto replicated_supplier_s_city_snapshot = ALUtils::take_snapshot(replicated_supplier.s_city);
    auto replicated_supplier_s_nation_snapshot = ALUtils::take_snapshot(replicated_supplier.s_nation);
    auto replicated_supplier_s_suppkey_snapshot = ALUtils::take_snapshot(replicated_supplier.s_suppkey);

    // Ensure local intermediate results allocation.
    numa_set_preferred(td->node);

    // Query program.
    auto C_76 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1997, 365);
    
    auto C_80 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_date_d_year_snapshot, 1998, 364);
    
    auto C_81 = merge_sorted<ps, uncompr_f, uncompr_f, uncompr_f >(C_76, C_80);
    
    auto X_82 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_datekey_snapshot, C_81);
    
    const column<uncompr_f> * X_85;
    const column<uncompr_f> * X_84;
    std::tie(X_85, X_84) = natural_equi_join_repl_t<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
    >::apply(
        X_82,
        replicated_lineorder_lo_orderdate_snapshot
    );
    
    
    auto X_91 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_partkey_snapshot, X_84);
    
    auto C_123 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_part_p_category_snapshot, 3, 31901);
    
    auto X_125 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_partkey_snapshot, C_123);
    
    const column<uncompr_f > * X_129;
    const column<uncompr_f > * X_128;
    std::tie(X_129, X_128) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_125,
        X_91,
        X_91->get_count_values()
    );
    
    
    auto X_136_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_suppkey_snapshot, X_84);
    X_136_0->template prepare_for_random_access<ps>();
    auto X_136 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_136_0, X_128);
    
    auto C_170 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_supplier_s_nation_snapshot, 23, 809);
    
    auto X_172 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_suppkey_snapshot, C_170);
    
    const column<uncompr_f > * X_176;
    const column<uncompr_f > * X_175;
    std::tie(X_176, X_175) = join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >(
        X_172,
        X_136,
        X_136->get_count_values()
    );
    
    
    auto X_181_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_custkey_snapshot, X_84);
    X_181_0->template prepare_for_random_access<ps>();
    auto X_181_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_181_0, X_128);
    X_181_1->template prepare_for_random_access<ps>();
    auto X_181 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_181_1, X_175);
    
    auto C_213 = my_select_repl_wit_t<equal, ps, uncompr_f>::apply(replicated_customer_c_region_snapshot, 1, 59761);
    
    auto X_215 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_customer_c_custkey_snapshot, C_213);
    
    auto X_217 = semi_join<
        ps,
        uncompr_f,
        uncompr_f,
        uncompr_f
        >
    (X_215, X_181);
    
    auto X_233_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_part_p_brand_snapshot, C_123);
    X_233_0->template prepare_for_random_access<ps>();
    auto X_233_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233_0, X_129);
    X_233_1->template prepare_for_random_access<ps>();
    auto X_233_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233_1, X_175);
    X_233_2->template prepare_for_random_access<ps>();
    auto X_233 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233_2, X_217);
    X_233->template prepare_for_random_access<ps>();
    
    auto X_235_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_supplier_s_city_snapshot, C_170);
    X_235_0->template prepare_for_random_access<ps>();
    auto X_235_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235_0, X_176);
    X_235_1->template prepare_for_random_access<ps>();
    auto X_235 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235_1, X_217);
    X_235->template prepare_for_random_access<ps>();
    
    auto X_230_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_date_d_year_snapshot, C_81);
    X_230_0->template prepare_for_random_access<ps>();
    auto X_230_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_230_0, X_85);
    X_230_1->template prepare_for_random_access<ps>();
    auto X_230_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_230_1, X_128);
    X_230_2->template prepare_for_random_access<ps>();
    auto X_230_3 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_230_2, X_175);
    X_230_3->template prepare_for_random_access<ps>();
    auto X_230 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_230_3, X_217);
    X_230->template prepare_for_random_access<ps>();
    
    const column<uncompr_f > * X_244;
    const column<uncompr_f > * C_245;
    std::tie(X_244, C_245) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f >(X_230);
    
    const column<uncompr_f > * X_247;
    const column<uncompr_f > * C_248;
    std::tie(X_247, C_248) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_244, X_235);
    
    const column<uncompr_f > * X_250;
    const column<uncompr_f > * C_251;
    std::tie(X_250, C_251) = group_vec<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f >(X_247, X_233);
    
    auto X_253 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_230, C_251);
    
    auto X_254 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_235, C_251);
    
    auto X_255 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_233, C_251);
    
    auto X_227_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_revenue_snapshot, X_84);
    X_227_0->template prepare_for_random_access<ps>();
    auto X_227_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_227_0, X_128);
    X_227_1->template prepare_for_random_access<ps>();
    auto X_227_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_227_1, X_175);
    X_227_2->template prepare_for_random_access<ps>();
    auto X_227 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_227_2, X_217);
    
    auto X_228_0 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_lineorder_lo_supplycost_snapshot, X_84);
    X_228_0->template prepare_for_random_access<ps>();
    auto X_228_1 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_228_0, X_128);
    X_228_1->template prepare_for_random_access<ps>();
    auto X_228_2 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_228_1, X_175);
    X_228_2->template prepare_for_random_access<ps>();
    auto X_228 = my_project_wit_t<ps, uncompr_f, uncompr_f, uncompr_f >::apply(X_228_2, X_217);
    
    auto X_241 = morphstore::calc_binary<sub, ps, uncompr_f, uncompr_f, uncompr_f >(X_227, X_228);
    
    // @todo Currently, the scalar processing style is hardcoded
    // in the query translation, because MorphStore still lacks a
    // vectorized implementation. As soon as such an
    // implementation exists, we should use it here.
    auto X_256 = agg_sum<scalar<v64<uint64_t>>, uncompr_f, uncompr_f, uncompr_f >(X_250, X_241, C_251->get_count_values());
    

    // Free all intermediate (and result) columns (including snapshots of replicated base data).
#ifdef MSV_NO_SELFMANAGED_MEMORY
    delete C_76;
    delete replicated_date_d_year_snapshot;
    delete C_80;
    delete C_81;
    delete X_82;
    delete replicated_date_d_datekey_snapshot;
    delete X_84;
    delete X_85;
    delete replicated_lineorder_lo_orderdate_snapshot;
    delete X_91;
    delete replicated_lineorder_lo_partkey_snapshot;
    delete C_123;
    delete replicated_part_p_category_snapshot;
    delete X_125;
    delete replicated_part_p_partkey_snapshot;
    delete X_128;
    delete X_129;
    delete X_136_0;
    delete replicated_lineorder_lo_suppkey_snapshot;
    delete X_136;
    delete C_170;
    delete replicated_supplier_s_nation_snapshot;
    delete X_172;
    delete replicated_supplier_s_suppkey_snapshot;
    delete X_175;
    delete X_176;
    delete X_181_0;
    delete replicated_lineorder_lo_custkey_snapshot;
    delete X_181_1;
    delete X_181;
    delete C_213;
    delete replicated_customer_c_region_snapshot;
    delete X_215;
    delete replicated_customer_c_custkey_snapshot;
    delete X_217;
    delete X_233_0;
    delete replicated_part_p_brand_snapshot;
    delete X_233_1;
    delete X_233_2;
    delete X_233;
    delete X_235_0;
    delete replicated_supplier_s_city_snapshot;
    delete X_235_1;
    delete X_235;
    delete X_230_0;
    delete X_230_1;
    delete X_230_2;
    delete X_230_3;
    delete X_230;
    delete C_245;
    delete X_244;
    delete C_248;
    delete X_247;
    delete C_251;
    delete X_250;
    delete X_253;
    delete X_254;
    delete X_255;
    delete X_227_0;
    delete replicated_lineorder_lo_revenue_snapshot;
    delete X_227_1;
    delete X_227_2;
    delete X_227;
    delete X_228_0;
    delete replicated_lineorder_lo_supplycost_snapshot;
    delete X_228_1;
    delete X_228_2;
    delete X_228;
    delete X_241;
    delete X_256;
#endif

    return;
}

// ****************************************************************************
// Function generating replicated columns from normal ones
// ****************************************************************************
inline void allocate_and_fill(replicated_column* rc, const column<uncompr_f>* column)
{
    rc = ALUtils::allocate(column->get_count_values() * sizeof(uint64_t) * 2, setting);

    append<scalar<v64<uint64_t>>>(rc, 12345665, 1024*1024*10);
    // Fill in replicated columns
    for (size_t k = 0; k < column->get_count_values()/2; k++)
    {
       append<scalar<v64<uint64_t>>>(rc, ((uint64_t*)column->get_data())[k], 1);
    }
}

// ****************************************************************************
// Main program
// ****************************************************************************
int main() {
    // Activate libnuma
    numa_available();

    // Activate PCM
    //m = PCM::getInstance();
    //PCM::ErrorCode status = m->program();
    //PCM::ErrorCode status2 = m->programServerUncoreMemoryMetrics(-1, -1, true);

    // Initialize vector of queries
    queries.push_back(&query_q11);
    queries.push_back(&query_q12);
    queries.push_back(&query_q13);
    queries.push_back(&query_q21);
    queries.push_back(&query_q22);
    queries.push_back(&query_q23);
    queries.push_back(&query_q31);
    queries.push_back(&query_q32);
    queries.push_back(&query_q33);
    queries.push_back(&query_q34);
    queries.push_back(&query_q41);
    queries.push_back(&query_q42);
    queries.push_back(&query_q43);

    // Replication sets
for (size_t c = 0; c < 8; c++)
{

    // ------------------------------------------------------------------------
    // Loading the base data
    // ------------------------------------------------------------------------
    std::cout << "Conf # " << c << std::endl;
    std::cout << "Loading the base data started... " << std::endl;

    // Allocate and fill the base data

    // Ensure base data local allocation
    numa_set_preferred(0);

    // Fill user given requirements
    setting.config = c;
    setting.replicaCount = 2;
    setting.isVolatileAllowed = false;
    setting.isCompressedAllowed = true;
    setting.isSequential = true;

    // Size of the base data in uint64_t integers
    //dataCount = 1000 * 1000 * 100;

    const std::string dataPath = "/root/morphstore/Benchmarks/ssb/data_sf1/cols_dict"; // TODO insert path to 'data_sf10/cols_dict/'
    // Load SSB base data from disk and load it into replicated columns.
    customer.c_city = binary_io<uncompr_f>::load(dataPath + "/customer.c_city.uncompr_f.bin");
    //allocate_and_fill(replicated_customer.c_city, customer.c_city);
    replicated_customer.c_city = ALUtils::allocate(customer.c_city->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < customer.c_city->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_customer.c_city, ((uint64_t*)customer.c_city->get_data())[k], 1);
    }
    delete customer.c_city;

    customer.c_custkey = binary_io<uncompr_f>::load(dataPath + "/customer.c_custkey.uncompr_f.bin");
    //allocate_and_fill(replicated_customer.c_custkey, customer.c_custkey);
    replicated_customer.c_custkey = ALUtils::allocate(customer.c_custkey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < customer.c_custkey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_customer.c_custkey, ((uint64_t*)customer.c_custkey->get_data())[k], 1);
    }
    delete customer.c_custkey;

    customer.c_nation = binary_io<uncompr_f>::load(dataPath + "/customer.c_nation.uncompr_f.bin");
    //allocate_and_fill(replicated_customer.c_nation, customer.c_nation);
    replicated_customer.c_nation = ALUtils::allocate(customer.c_nation->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < customer.c_nation->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_customer.c_nation, ((uint64_t*)customer.c_nation->get_data())[k], 1);
    }
    delete customer.c_nation;

    customer.c_region = binary_io<uncompr_f>::load(dataPath + "/customer.c_region.uncompr_f.bin");
    //allocate_and_fill(replicated_customer.c_region, customer.c_region);
    replicated_customer.c_region = ALUtils::allocate(customer.c_region->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < customer.c_region->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_customer.c_region, ((uint64_t*)customer.c_region->get_data())[k], 1);
    }
    delete customer.c_region;

    date.d_datekey = binary_io<uncompr_f>::load(dataPath + "/date.d_datekey.uncompr_f.bin");
    //allocate_and_fill(replicated_date.d_datekey, date.d_datekey);
    replicated_date.d_datekey = ALUtils::allocate(date.d_datekey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < date.d_datekey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_date.d_datekey, ((uint64_t*)date.d_datekey->get_data())[k], 1);
    }
    delete date.d_datekey;

    date.d_weeknuminyear = binary_io<uncompr_f>::load(dataPath + "/date.d_weeknuminyear.uncompr_f.bin");
    //allocate_and_fill(replicated_date.d_weeknuminyear, date.d_weeknuminyear);
    replicated_date.d_weeknuminyear = ALUtils::allocate(date.d_weeknuminyear->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < date.d_weeknuminyear->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_date.d_weeknuminyear, ((uint64_t*)date.d_weeknuminyear->get_data())[k], 1);
    }
    delete date.d_weeknuminyear;

    date.d_year = binary_io<uncompr_f>::load(dataPath + "/date.d_year.uncompr_f.bin");
    //allocate_and_fill(replicated_date.d_year, date.d_year);
    replicated_date.d_year = ALUtils::allocate(date.d_year->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < date.d_year->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_date.d_year, ((uint64_t*)date.d_year->get_data())[k], 1);
    }
    delete date.d_year;

    date.d_yearmonth = binary_io<uncompr_f>::load(dataPath + "/date.d_yearmonth.uncompr_f.bin");
    //allocate_and_fill(replicated_date.d_yearmonth, date.d_yearmonth);
    replicated_date.d_yearmonth = ALUtils::allocate(date.d_yearmonth->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < date.d_yearmonth->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_date.d_yearmonth, ((uint64_t*)date.d_yearmonth->get_data())[k], 1);
    }
    delete date.d_yearmonth;

    date.d_yearmonthnum = binary_io<uncompr_f>::load(dataPath + "/date.d_yearmonthnum.uncompr_f.bin");
    //allocate_and_fill(replicated_date.d_yearmonthnum, date.d_yearmonthnum);
    replicated_date.d_yearmonthnum = ALUtils::allocate(date.d_yearmonthnum->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < date.d_yearmonthnum->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_date.d_yearmonthnum, ((uint64_t*)date.d_yearmonthnum->get_data())[k], 1);
    }
    delete date.d_yearmonthnum;

    lineorder.lo_custkey = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_custkey.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_custkey, lineorder.lo_custkey);
    replicated_lineorder.lo_custkey = ALUtils::allocate(lineorder.lo_custkey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_custkey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_custkey, ((uint64_t*)lineorder.lo_custkey->get_data())[k], 1);
    }
    delete lineorder.lo_custkey;

    lineorder.lo_discount = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_discount.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_discount, lineorder.lo_discount);
    replicated_lineorder.lo_discount = ALUtils::allocate(lineorder.lo_discount->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_discount->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_discount, ((uint64_t*)lineorder.lo_discount->get_data())[k], 1);
    }
    delete lineorder.lo_discount;

    lineorder.lo_extendedprice = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_extendedprice.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_extendedprice, lineorder.lo_extendedprice);
    replicated_lineorder.lo_extendedprice = ALUtils::allocate(lineorder.lo_extendedprice->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_extendedprice->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_extendedprice, ((uint64_t*)lineorder.lo_extendedprice->get_data())[k], 1);
    }
    delete lineorder.lo_extendedprice;

    lineorder.lo_orderdate = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_orderdate.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_orderdate, lineorder.lo_orderdate);
    replicated_lineorder.lo_orderdate = ALUtils::allocate(lineorder.lo_orderdate->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_orderdate->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_orderdate, ((uint64_t*)lineorder.lo_orderdate->get_data())[k], 1);
    }
    delete lineorder.lo_orderdate;

    lineorder.lo_partkey = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_partkey.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_partkey, lineorder.lo_partkey);
    replicated_lineorder.lo_partkey = ALUtils::allocate(lineorder.lo_partkey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_partkey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_partkey, ((uint64_t*)lineorder.lo_partkey->get_data())[k], 1);
    }
    delete lineorder.lo_partkey;

    lineorder.lo_quantity = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_quantity.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_quantity, lineorder.lo_quantity);
    replicated_lineorder.lo_quantity = ALUtils::allocate(lineorder.lo_quantity->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_quantity->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_quantity, ((uint64_t*)lineorder.lo_quantity->get_data())[k], 1);
    }
    delete lineorder.lo_quantity;

    lineorder.lo_revenue = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_revenue.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_revenue, lineorder.lo_revenue);
    replicated_lineorder.lo_revenue = ALUtils::allocate(lineorder.lo_revenue->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_revenue->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_revenue, ((uint64_t*)lineorder.lo_revenue->get_data())[k], 1);
    }
    delete lineorder.lo_revenue;

    lineorder.lo_suppkey = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_suppkey.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_suppkey, lineorder.lo_suppkey);
    replicated_lineorder.lo_suppkey = ALUtils::allocate(lineorder.lo_suppkey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_suppkey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_suppkey, ((uint64_t*)lineorder.lo_suppkey->get_data())[k], 1);
    }
    delete lineorder.lo_suppkey;

    lineorder.lo_supplycost = binary_io<uncompr_f>::load(dataPath + "/lineorder.lo_supplycost.uncompr_f.bin");
    //allocate_and_fill(replicated_lineorder.lo_supplycost, lineorder.lo_supplycost);
    replicated_lineorder.lo_supplycost = ALUtils::allocate(lineorder.lo_supplycost->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < lineorder.lo_supplycost->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_lineorder.lo_supplycost, ((uint64_t*)lineorder.lo_supplycost->get_data())[k], 1);
    }
    delete lineorder.lo_supplycost;

    part.p_brand = binary_io<uncompr_f>::load(dataPath + "/part.p_brand.uncompr_f.bin");
    //allocate_and_fill(replicated_part.p_brand, part.p_brand);
    replicated_part.p_brand = ALUtils::allocate(part.p_brand->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < part.p_brand->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_part.p_brand, ((uint64_t*)part.p_brand->get_data())[k], 1);
    }
    delete part.p_brand;

    part.p_category = binary_io<uncompr_f>::load(dataPath + "/part.p_category.uncompr_f.bin");
    //allocate_and_fill(replicated_part.p_category, part.p_category);
    replicated_part.p_category = ALUtils::allocate(part.p_category->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < part.p_category->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_part.p_category, ((uint64_t*)part.p_category->get_data())[k], 1);
    }
    delete part.p_category;

    part.p_mfgr = binary_io<uncompr_f>::load(dataPath + "/part.p_mfgr.uncompr_f.bin");
    //allocate_and_fill(replicated_part.p_mfgr, part.p_mfgr);
    replicated_part.p_mfgr = ALUtils::allocate(part.p_mfgr->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < part.p_mfgr->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_part.p_mfgr, ((uint64_t*)part.p_mfgr->get_data())[k], 1);
    }
    delete part.p_mfgr;

    part.p_partkey = binary_io<uncompr_f>::load(dataPath + "/part.p_partkey.uncompr_f.bin");
    //allocate_and_fill(replicated_part.p_partkey, part.p_partkey);
    replicated_part.p_partkey = ALUtils::allocate(part.p_partkey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < part.p_partkey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_part.p_partkey, ((uint64_t*)part.p_partkey->get_data())[k], 1);
    }
    delete part.p_partkey;

    supplier.s_city = binary_io<uncompr_f>::load(dataPath + "/supplier.s_city.uncompr_f.bin");
    //allocate_and_fill(replicated_supplier.s_city, supplier.s_city);
    replicated_supplier.s_city = ALUtils::allocate(supplier.s_city->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < supplier.s_city->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_supplier.s_city, ((uint64_t*)supplier.s_city->get_data())[k], 1);
    }
    delete supplier.s_city;

    supplier.s_nation = binary_io<uncompr_f>::load(dataPath + "/supplier.s_nation.uncompr_f.bin");
    //allocate_and_fill(replicated_supplier.s_nation, supplier.s_nation);
    replicated_supplier.s_nation = ALUtils::allocate(supplier.s_nation->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < supplier.s_nation->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_supplier.s_nation, ((uint64_t*)supplier.s_nation->get_data())[k], 1);
    }
    delete supplier.s_nation;

    supplier.s_region = binary_io<uncompr_f>::load(dataPath + "/supplier.s_region.uncompr_f.bin");
    //allocate_and_fill(replicated_supplier.s_region, supplier.s_region);
    replicated_supplier.s_region = ALUtils::allocate(supplier.s_region->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < supplier.s_region->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_supplier.s_region, ((uint64_t*)supplier.s_region->get_data())[k], 1);
    }
    delete supplier.s_region;

    supplier.s_suppkey = binary_io<uncompr_f>::load(dataPath + "/supplier.s_suppkey.uncompr_f.bin");
    //allocate_and_fill(replicated_supplier.s_suppkey, supplier.s_suppkey);
    replicated_supplier.s_suppkey = ALUtils::allocate(supplier.s_suppkey->get_count_values() * sizeof(uint64_t), setting);
    for (size_t k = 0; k < supplier.s_suppkey->get_count_values(); k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_supplier.s_suppkey, ((uint64_t*)supplier.s_suppkey->get_data())[k], 1);
    }
    delete supplier.s_suppkey;

    std::cout << "done." << std::endl;

    // Queries
  for (size_t l = 0; l < 1; l++)
  {
    std::cout << "Query num: " << l << std::endl;

    std::cout << "Cyclic query execution on the same base data," << std::endl << "started, from 1 to " << countThreads << " thread(s), runtime: " << measuringTimeMillis << "ms per experiment... " << std::endl;
    //std::cout << "Threads;RunsPerSecond;CumulativeRuntime;RelativeSpeedup;RelativeSlowdown;TotalRunsDone" << std::endl;
    std::cout << "Threads;RunsPerSecond;Speedup;TotalRuns" << std::endl;


    double singleThreadRuntime = 0;
    double singleThreadRunsPerSecond = 0;

    for (size_t j = 1; j < countThreads+1; )
    //for (size_t j = 1; j < countThreads+1; j += 23)
    //for (size_t j = 1; j < countThreads+1; j++)
    {

    // ------------------------------------------------------------------------
    // Parallel query execution
    // ------------------------------------------------------------------------

    // Reset statistcs
    for (size_t i = 0; i < 13; i++)
    {
        counters[i] = 0;
    }

    waitingThreads = j;
    //waitingThreads = j+1;
    std::vector<ThreadData> threadParameters(j);
    // Memory bandwidth snooping thread
    //ThreadData memSnoopThread;
    //memSnoopThread.core = 95;
    //memSnoopThread.nodeSize = numa_num_configured_cpus() / (numa_max_node() + 1) / 2; //if HT is disabled do not divide by 2

    // Specify theads affiliation
    for (size_t i = 0; i < j; i++)
    {
        threadParameters[i].core = i;
        threadParameters[i].nodeSize = numa_num_configured_cpus() / (numa_max_node() + 1) / 2; //if HT is disabled do not divide by 2

        threadParameters[i].queryNum = l;
    }
    //std::cout << "Threads:" << j << std::endl;
    // Create all threads
    for (size_t i = 0; i < j; i++)
    {
        pthread_create(&threadParameters[i].thread, nullptr, query_executor, &threadParameters[i]);
    }
    // Memory bandwidth snooping thread
    //pthread_create(&memSnoopThread.thread, nullptr, updater, &memSnoopThread);

    std::this_thread::sleep_for(std::chrono::milliseconds(measuringTimeMillis));

    // Synchronization to stop at a same time
    for (size_t i = 0; i < j; i++)
    {
        threadParameters[i].halt = true;
    }
    //memSnoopThread.halt = true;

    for (size_t i = 0; i < j; i++)
    {
        pthread_join(threadParameters[i].thread, nullptr);
    }
    //pthread_join(memSnoopThread.thread, nullptr);

    // Statistical calculation
    double cumulativeRuntime = 0;
    double totalRunsExecuted = 0;
    double totalRunsPerSecond = 0;
    for (size_t i = 0; i < j; i++)
    {
        cumulativeRuntime += threadParameters[i].sw.duration();
        totalRunsExecuted += threadParameters[i].runsExecuted;
        totalRunsPerSecond += threadParameters[i].runsExecuted / threadParameters[i].sw.duration();
    }

    if (j == 1)
    {
        singleThreadRuntime = cumulativeRuntime / totalRunsExecuted;
        singleThreadRunsPerSecond = totalRunsPerSecond;
    }
    //std::cout << j << ";" << totalRunsPerSecond << ";" << cumulativeRuntime / totalRunsExecuted << ";" << totalRunsPerSecond / singleThreadRunsPerSecond << ";" << singleThreadRuntime / (cumulativeRuntime / totalRunsExecuted) << ";" << totalRunsExecuted << std::endl;

    size_t throughput = 0;
    for (size_t i = 0; i < QUERIES_NUM; i++)
    {
        //std::cout << i + 1 <<" query executions: " << counters[i] << std::endl;
        throughput += counters[i];
    }

    std::cout << j << ";" << totalRunsPerSecond << ";" << totalRunsPerSecond / singleThreadRunsPerSecond << ";" << throughput << std::endl;

    if (j == 1)
      j = 3;
    else if (j == 3)
      j = 6;
    else if (j == 6)
      j = 12;
    else if (j == 12)
      j = 24;
    else if (j == 24)
      j = 48;
    else if (j == 48)
      j = 72;
    else if (j == 72)
      j = 96;
    else
      j = 200;

    //if (j < 4)
    //  j++;
    //else
    //  j += 4;

    }
  }  // End queries loop

    // Free replicated base data.
    delete(replicated_customer.c_city);
    delete(replicated_customer.c_custkey);
    delete(replicated_customer.c_nation);
    delete(replicated_customer.c_region);
    delete(replicated_date.d_datekey);
    delete(replicated_date.d_weeknuminyear);
    delete(replicated_date.d_year);
    delete(replicated_date.d_yearmonth);
    delete(replicated_date.d_yearmonthnum);
    delete(replicated_lineorder.lo_custkey);
    delete(replicated_lineorder.lo_discount);
    delete(replicated_lineorder.lo_extendedprice);
    delete(replicated_lineorder.lo_orderdate);
    delete(replicated_lineorder.lo_partkey);
    delete(replicated_lineorder.lo_quantity);
    delete(replicated_lineorder.lo_revenue);
    delete(replicated_lineorder.lo_suppkey);
    delete(replicated_lineorder.lo_supplycost);
    delete(replicated_part.p_brand);
    delete(replicated_part.p_category);
    delete(replicated_part.p_mfgr);
    delete(replicated_part.p_partkey);
    delete(replicated_supplier.s_city);
    delete(replicated_supplier.s_nation);
    delete(replicated_supplier.s_region);
    delete(replicated_supplier.s_suppkey);
} // End replication sets loop

    return 0;
}