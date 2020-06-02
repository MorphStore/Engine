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
#define QUERIES_NUM 5

// PCM counters
//PCM* m;

std::atomic<size_t> waitingThreads; // = {countThreads};

struct ThreadData {
    pthread_t thread;
    size_t node = 0;
    size_t core = 0;
    size_t nodeSize = 0;
    bool halt = false;

    //replicated_column* replicated_datat = nullptr;  // Per thread column
    column<uncompr_f> * indecest = nullptr;
    column<uncompr_f> * indecest1000 = nullptr;

    size_t runsExecuted = 0;
    WallClockStopWatch sw;

    size_t queryNum = 0;
};

// ****************************************************************************
// Base data declaration
// ****************************************************************************

replicated_column* replicated_data;
replicated_column* replicated_data2;
replicated_column* replicated_indeces;
column<uncompr_f> * data;
column<uncompr_f> * data2;
column<uncompr_f> * indeces;
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

void query1(ThreadData* td);
void query2(ThreadData* td);
void query3(ThreadData* td);
void query4(ThreadData* td);
void query5(ThreadData* td);

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
    if (td->core < td->nodeSize || td->core >= td->nodeSize * 3)
        CPU_SET(td->core, &cpuset);
    else if (td->core < td->nodeSize*2)
        CPU_SET(td->core + td->nodeSize, &cpuset);
    else
        CPU_SET(td->core - td->nodeSize, &cpuset);

//    CPU_SET(td->core, &cpuset);
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

    // Local memory allocation
    //td->replicated_datat = ALUtils::allocate(dataCount, setting);
    //for (size_t k = 0; k < dataCount / sizeof(size_t); k++)
    //{
    //   append<scalar<v64<uint64_t>>>(td->replicated_datat, ((uint32_t*)data->get_data())[k], 1);
    //}
    //td->indecest = generate_with_distr(dataCount / 10, std::uniform_int_distribution<uint64_t>(0, dataCount-1), false);
    //td->indecest1000 = generate_with_distr(dataCount, std::uniform_int_distribution<uint64_t>(0, (dataCount-1)/10), false);

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
    std::this_thread::sleep_for(std::chrono::milliseconds((dis(gen)*100) + ((dis(gen))+5)*100));

    td->sw.start();

    while (!td->halt)
    {
       // Pick up randomly one query for execution
       size_t random = td->queryNum; //dis(gen); //td->queryNum;
       queries[random](td);
       counters[random]++;
       td->runsExecuted++;
    }

    td->sw.stop();

    //delete td->replicated_datat;
    //delete td->indecest;
    //delete td->indecest1000;
    return nullptr;
}

void query1(ThreadData* td)
{
    // Aggregation
    // Query program.
    using ps = scalar<v64<uint64_t>>;

    auto replicated_data_snapshot = ALUtils::take_snapshot(replicated_data);

    // Ensure local intermidiate results allocation
    numa_set_preferred(td->node);

    //auto val = sequential_read<ps, uncompr_f>(datat);
    //auto val = sequential_read<ps, uncompr_f>(data);
    //auto val = random_read<ps, uncompr_f>(data, indeces);
    //auto val = sequential_write<ps, uncompr_f>(data);
    //auto val = random_write<ps, uncompr_f>(data, indeces);
    auto res = agg_sum_repl_t<ps>::apply(replicated_data_snapshot);

    //print_columns(print_buffer_base::decimal, res, "SUM(baseCol2)");

    delete replicated_data_snapshot;
    delete res;

    return;
}

void query2(ThreadData* td)
{
    // Selection 10%
    // Query program.
    using ps = scalar<v64<uint64_t>>;

    //auto replicated_data_snapshot = ALUtils::take_snapshot(replicated_data);

    // Ensure local intermidiate results allocation
    //numa_set_localalloc();
    numa_set_strict(td->node);
    //numa_set_preferred(td->node);
    //std::cout << "No: "<<td->node<<std::endl;
    //auto val = sequential_read<ps, uncompr_f>(datat);
    //auto val = sequential_read<ps, uncompr_f>(data);
    //auto val = random_read<ps, uncompr_f>(data, indeces);
    //auto val = sequential_write<ps, uncompr_f>(data);
    //auto val = random_write<ps, uncompr_f>(data, indeces);
    //auto res = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_data, 1);
    //auto res = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_data, indeces);
    //auto res = agg_sum_repl_t<ps>::apply(replicated_data);

    //auto res = my_select_repl_wit_t<vectorlib::less, ps, uncompr_f>::apply(replicated_data_snapshot, dataCount/10, dataCount/8);
     if (td->node == 0)
     {
        auto res = morphstore::select<std::less, ps, uncompr_f, uncompr_f>(data, dataCount/10);
        delete res;
     }
     else
     {
        auto res = morphstore::select<std::less, ps, uncompr_f, uncompr_f>(data2, dataCount/10);
        delete res;
     }

    //auto res = my_select_repl_wit_t<vectorlib::less, ps, uncompr_f>::apply(replicated_data, dataCount/10, dataCount/8);

    //auto res = semi_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f>::apply(replicated_data, indeces);
    //auto res = natural_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>::apply(replicated_data_snapshot, indeces);
    //auto res = semi_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f>::apply(indeces, replicated_data);
    //auto res = natural_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>::apply(indeces, replicated_data);
    //auto res = agg_sum_repl_t<ps>::apply(replicated_data);

    //print_columns(print_buffer_base::decimal, res, "SUM(baseCol2)");

    //delete std::get<0>(res);
    //delete std::get<1>(res);
    //delete res;
    //delete replicated_data_snapshot;

    return;
}

void query3(ThreadData* td)
{
    // Selection 1%
    // Query program.
    using ps = scalar<v64<uint64_t>>;
    auto replicated_data_snapshot = ALUtils::take_snapshot(replicated_data);

    // Ensure local intermidiate results allocation
    numa_set_preferred(td->node);

    //auto replicated_indeces_snapshot = ALUtils::take_snapshot(replicated_indeces);
    //auto val = sequential_read<ps, uncompr_f>(datat);
    //auto val = sequential_read<ps, uncompr_f>(data);
    //auto val = random_read<ps, uncompr_f>(data, indeces);
    //auto val = sequential_write<ps, uncompr_f>(data);
    //auto val = random_write<ps, uncompr_f>(data, indeces);
    //auto res = my_select_repl_wit_t<greaterequal, ps, uncompr_f>::apply(replicated_data, 1);
    //auto res = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_data, indeces);
    //auto res = agg_sum_repl_t<ps>::apply(replicated_data);

    auto res = my_select_repl_wit_t<vectorlib::less, ps, uncompr_f>::apply(replicated_data_snapshot, dataCount/10, dataCount/8);

    //auto res = semi_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f>::apply(replicated_data, indeces);
    //auto res = semi_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f>::apply(replicated_data, indeces);
    //auto res = natural_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>::apply(replicated_data, indeces);
    //auto res = semi_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f>::apply(indeces, replicated_data);
    //auto res = natural_equi_join_repl_t<ps, uncompr_f, uncompr_f, uncompr_f, uncompr_f>::apply(indeces, replicated_data);
    //auto res = agg_sum_repl_t<ps>::apply(replicated_data);

    //print_columns(print_buffer_base::decimal, res, "SUM(baseCol2)");

    //delete std::get<0>(res);
    //delete std::get<1>(res);
    delete res;
    delete replicated_data_snapshot;

    //delete replicated_indeces_snapshot;
    return;
}

void query4(ThreadData* td)
{
    // Projection 10% index column size
    // Query program.
    using ps = scalar<v64<uint64_t>>;

    auto replicated_data_snapshot = ALUtils::take_snapshot(replicated_data);

    //auto replicated_indeces_snapshot = ALUtils::take_snapshot(replicated_indeces);
/*
    // Positions fulfilling "replicated_data >= 150"
    auto i1 = morphstore::my_select_repl_wit_t<
            greaterequal,
            ps,
            uncompr_f
    >::apply(replicated_data_snapshot, 150);

    // Data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto i2 = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_data_snapshot, i1);

    // Sum over the data elements of "baseCol2" fulfilling "baseCol1 = 150"
    auto res = agg_sum_t<ps, uncompr_f>::apply(i2);

    //print_columns(print_buffer_base::decimal, res, "SUM(baseCol2)");
    delete i1;
    delete i2;
*/
    auto res = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_data_snapshot, td->indecest);

    delete replicated_data_snapshot;
    delete res;

    return;
}

void query5(ThreadData* td)
{
    // Projection 1000% index column size
    // Query program.
    using ps = scalar<v64<uint64_t>>;

    //auto val = sequential_read<ps, uncompr_f>(datat);
    //auto val = sequential_read<ps, uncompr_f>(data);
    //auto val = random_read<ps, uncompr_f>(data, indeces);
    //auto val = sequential_write<ps, uncompr_f>(data);
    //auto val = random_write<ps, uncompr_f>(data, indeces);
    //auto res = agg_sum_repl_t<ps>::apply(replicated_data);

    //append<ps>(replicated_data, 1, 8);
    //print_columns(print_buffer_base::decimal, res, "SUM(baseCol2)");

    auto replicated_data_snapshot = ALUtils::take_snapshot(replicated_data2);

    auto res = my_project_repl_wit_t<ps, uncompr_f, uncompr_f>::apply(replicated_data_snapshot, td->indecest1000);

    delete replicated_data_snapshot;
    delete res;

    return;
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
    queries.push_back(&query1);
    queries.push_back(&query2);
    queries.push_back(&query3);
    queries.push_back(&query4);
    queries.push_back(&query5);

    // Replication sets
for (size_t c = 3; c < 4; c++)
{

    // ------------------------------------------------------------------------
    // Loading the base data
    // ------------------------------------------------------------------------
    std::cout << "Conf # " << c << std::endl;
    std::cout << "Loading the base data started... " << std::endl;

    // Allocate and fill the base data

    // Ensure base data local allocation
    //numa_set_localalloc();
    numa_set_preferred(0);

    // Fill user given requirements
    setting.config = c;
    setting.replicaCount = 2;
    setting.isVolatileAllowed = false;
    setting.isCompressedAllowed = true;
    setting.isSequential = true;

    // Size of the base data in uint64_t integers
    dataCount = 1000 * 1000 * 100;

    replicated_data = ALUtils::allocate(dataCount * sizeof(uint64_t) * 2, setting);
    replicated_data2 = ALUtils::allocate(dataCount * sizeof(uint64_t) / 10, setting);
    //replicated_indeces = ALUtils::allocate(dataCount * sizeof(uint64_t), setting);
numa_set_preferred(0);
    data = generate_with_distr(dataCount, std::uniform_int_distribution<uint64_t>(0, dataCount-1), false);
numa_set_preferred(1);
    data2 = generate_with_distr(dataCount, std::uniform_int_distribution<uint64_t>(0, dataCount-1), false);
numa_set_preferred(0);
    //indeces = generate_with_distr(dataCount, std::uniform_int_distribution<uint64_t>(0, dataCount-1), false);

    //append<scalar<v64<uint64_t>>>(replicated_data, 1000, dataCount / sizeof(size_t));
    //append<scalar<v64<uint64_t>>>(replicated_data2, 1000, dataCount / sizeof(size_t) / 10);
    //append<scalar<v64<uint64_t>>>(replicated_indeces, 1, dataCount / sizeof(size_t)  / 32);

    // Fill in replicated columns
    for (size_t k = 0; k < dataCount; k++)
    {
       append<scalar<v64<uint64_t>>>(replicated_data, ((uint64_t*)data->get_data())[k], 1);
       //append<scalar<v64<uint64_t>>>(replicated_indeces, ((uint64_t*)indeces->get_data())[k], 1);
    }

    // Fill in replicated columns 2
    for (size_t k = 0; k < dataCount / 10; k++)
    {
       //append<scalar<v64<uint64_t>>>(replicated_data2, ((uint64_t*)data->get_data())[k], 1);
       //append<scalar<v64<uint64_t>>>(replicated_indeces, ((uint64_t*)indeces->get_data())[k], 1);
    }
    std::cout << "done." << std::endl;

    // Queries
  for (size_t l = 1; l < 2; l++)
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
    for (size_t i = 0; i < 5; i++)
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
      j = 4;
    else if (j == 4)
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
  }  //End queries loop

    delete data;
    delete data2;
    //delete indeces;
    delete replicated_data;
    delete replicated_data2;
    //delete replicated_indeces;
} //End replication sets loop

    return 0;
}