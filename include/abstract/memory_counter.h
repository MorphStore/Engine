#define HACK_TO_REMOVE_DUPLICATE_ERROR
#include <iostream>
#ifdef _MSC_VER
#include <windows.h>
#include "../../../../pcm/PCM_Win/windriver.h"
#else
#include <unistd.h>
#include <signal.h>
#include <sys/time.h> // for gettimeofday()
#endif
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <assert.h>
#include "../../../../pcm/cpucounters.h"
#include "../../../../pcm/utils.h"

//Programmable iMC counter
#define READ 0
#define WRITE 1
#define READ_RANK_A 0
#define WRITE_RANK_A 1
#define READ_RANK_B 2
#define WRITE_RANK_B 3
#define PARTIAL 2
#define PMM_READ 2
#define PMM_WRITE 3
#define NM_HIT 0  // NM :  Near Memory (DRAM cache) in Memory Mode
#define PCM_DELAY_DEFAULT 1.0 // in seconds
#define PCM_DELAY_MIN 0.015 // 15 milliseconds is practical on most modern CPUs
#define PCM_CALIBRATION_INTERVAL 50 // calibrate clock only every 50th iteration

#define DEFAULT_DISPLAY_COLUMNS 2

using namespace std;

const uint32 max_sockets = 256;
const uint32 max_imc_channels = ServerUncorePowerState::maxChannels;
const uint32 max_edc_channels = ServerUncorePowerState::maxChannels;
const uint32 max_imc_controllers = ServerUncorePowerState::maxControllers;

typedef struct memdata {
    float iMC_Rd_socket_chan[max_sockets][max_imc_channels];
    float iMC_Wr_socket_chan[max_sockets][max_imc_channels];
    float iMC_PMM_Rd_socket_chan[max_sockets][max_imc_channels];
    float iMC_PMM_Wr_socket_chan[max_sockets][max_imc_channels];
    float iMC_Rd_socket[max_sockets];
    float iMC_Wr_socket[max_sockets];
    float iMC_PMM_Rd_socket[max_sockets];
    float iMC_PMM_Wr_socket[max_sockets];
    float M2M_NM_read_hit_rate[max_sockets][max_imc_controllers];
    float EDC_Rd_socket_chan[max_sockets][max_edc_channels];
    float EDC_Wr_socket_chan[max_sockets][max_edc_channels];
    float EDC_Rd_socket[max_sockets];
    float EDC_Wr_socket[max_sockets];
    uint64 partial_write[max_sockets];
    bool PMM;
} memdata_t;

void print_help(const string prog_name)
{
    cerr << endl << " Usage: " << endl << " " << prog_name
         << " --help | [delay] [options] [-- external_program [external_program_options]]" << endl;
    cerr << "   <delay>                           => time interval to sample performance counters." << endl;
    cerr << "                                        If not specified, or 0, with external program given" << endl;
    cerr << "                                        will read counters only after external program finishes" << endl;
    cerr << " Supported <options> are: " << endl;
    cerr << "  -h    | --help  | /h               => print this help and exit" << endl;
    cerr << "  -rank=X | /rank=X                  => monitor DIMM rank X. At most 2 out of 8 total ranks can be monitored simultaneously." << endl;
    cerr << "  -pmm                               => monitor PMM memory bandwidth (instead of partial writes)." << endl;
    cerr << "  -nc   | --nochannel | /nc          => suppress output for individual channels." << endl;
    cerr << "  -csv[=file.csv] | /csv[=file.csv]  => output compact CSV format to screen or" << endl
         << "                                        to a file, in case filename is provided" << endl;
    cerr << "  -columns=X | /columns=X            => Number of columns to display the NUMA Nodes, defaults to 2." << endl;
#ifdef _MSC_VER
    cerr << "  --uninstallDriver | --installDriver=> (un)install driver" << endl;
#endif
    cerr << " Examples:" << endl;
    cerr << "  " << prog_name << " 1                  => print counters every second without core and socket output" << endl;
    cerr << "  " << prog_name << " 0.5 -csv=test.log  => twice a second save counter values to test.log in CSV format" << endl;
    cerr << "  " << prog_name << " /csv 5 2>/dev/null => one sampe every 5 seconds, and discard all diagnostic output" << endl;
    cerr << endl;
}

void printSocketBWHeader(uint32 no_columns, uint32 skt, const bool show_channel_output)
{
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|---------------------------------------|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|--             Socket "<<setw(2)<<i<<"             --|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|---------------------------------------|";
    }
    cout << endl;
    if (show_channel_output) {
       for (uint32 i=skt; i<(no_columns+skt); ++i) {
           cout << "|--     Memory Channel Monitoring     --|";
       }
       cout << endl;
       for (uint32 i=skt; i<(no_columns+skt); ++i) {
           cout << "|---------------------------------------|";
       }
       cout << endl;
    }
}

void printSocketRankBWHeader(uint32 no_columns, uint32 skt)
{
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|-------------------------------------------|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|--               Socket "<<setw(2)<<i<<"               --|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|-------------------------------------------|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|--           DIMM Rank Monitoring        --|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|-------------------------------------------|";
    }
    cout << endl;
}

void printSocketChannelBW(PCM *m, memdata_t *md, uint32 no_columns, uint32 skt)
{
    for (uint32 channel = 0; channel < max_imc_channels; ++channel) {
        // check all the sockets for bad channel "channel"
        unsigned bad_channels = 0;
        for (uint32 i=skt; i<(skt+no_columns); ++i) {
            if (md->iMC_Rd_socket_chan[i][channel] < 0.0 || md->iMC_Wr_socket_chan[i][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
                ++bad_channels;
        }
        if (bad_channels == no_columns) { // the channel is missing on all sockets in the row
            continue;
        }
        for (uint32 i=skt; i<(skt+no_columns); ++i) {
            cout << "|-- Mem Ch "<<setw(2)<<channel<<": Reads (MB/s): "<<setw(8)<<md->iMC_Rd_socket_chan[i][channel]<<" --|";
        }
        cout << endl;
        for (uint32 i=skt; i<(skt+no_columns); ++i) {
            cout << "|--            Writes(MB/s): "<<setw(8)<<md->iMC_Wr_socket_chan[i][channel]<<" --|";
        }
        cout << endl;
        if(md->PMM)
        {
            for (uint32 i=skt; i<(skt+no_columns); ++i) {
                cout << "|--      PMM Reads(MB/s)   : "<<setw(8)<<md->iMC_PMM_Rd_socket_chan[i][channel]<<" --|";
            }
            cout << endl;
            for (uint32 i=skt; i<(skt+no_columns); ++i) {
                cout << "|--      PMM Writes(MB/s)  : "<<setw(8)<<md->iMC_PMM_Wr_socket_chan[i][channel]<<" --|";
            }
            cout << endl;
        }
    }
}

void printSocketChannelBW(uint32 no_columns, uint32 skt, uint32 num_imc_channels, const ServerUncorePowerState * uncState1, const ServerUncorePowerState * uncState2, uint64 elapsedTime, int rankA, int rankB)
{
    for (uint32 channel = 0; channel < num_imc_channels; ++channel) {
        if(rankA >= 0) {
          for (uint32 i=skt; i<(skt+no_columns); ++i) {
              cout << "|-- Mem Ch "<<setw(2)<<channel<<" R " << setw(1) << rankA <<": Reads (MB/s): "<<setw(8)<<(float) (getMCCounter(channel,READ_RANK_A,uncState1[i],uncState2[i]) * 64 / 1000000.0 / (elapsedTime/1000.0))<<" --|";
          }
          cout << endl;
          for (uint32 i=skt; i<(skt+no_columns); ++i) {
              cout << "|--                Writes(MB/s): "<<setw(8)<<(float) (getMCCounter(channel,WRITE_RANK_A,uncState1[i],uncState2[i]) * 64 / 1000000.0 / (elapsedTime/1000.0))<<" --|";
          }
          cout << endl;
        }
        if(rankB >= 0) {
          for (uint32 i=skt; i<(skt+no_columns); ++i) {
              cout << "|-- Mem Ch "<<setw(2) << channel<<" R " << setw(1) << rankB <<": Reads (MB/s): "<<setw(8)<<(float) (getMCCounter(channel,READ_RANK_B,uncState1[i],uncState2[i]) * 64 / 1000000.0 / (elapsedTime/1000.0))<<" --|";
          }
          cout << endl;
          for (uint32 i=skt; i<(skt+no_columns); ++i) {
              cout << "|--                Writes(MB/s): "<<setw(8)<<(float) (getMCCounter(channel,WRITE_RANK_B,uncState1[i],uncState2[i]) * 64 / 1000000.0 / (elapsedTime/1000.0))<<" --|";
          }
          cout << endl;
        }
    }
}

void printSocketBWFooter(uint32 no_columns, uint32 skt, const memdata_t *md)
{
    for (uint32 i=skt; i<(skt+no_columns); ++i) {
        cout << "|-- NODE"<<setw(2)<<i<<" Mem Read (MB/s) : "<<setw(8)<<md->iMC_Rd_socket[i]<<" --|";
    }
    cout << endl;
    for (uint32 i=skt; i<(skt+no_columns); ++i) {
        cout << "|-- NODE"<<setw(2)<<i<<" Mem Write(MB/s) : "<<setw(8)<<md->iMC_Wr_socket[i]<<" --|";
    }
    cout << endl;
    if (md->PMM)
    {
        for (uint32 i=skt; i<(skt+no_columns); ++i) {
            cout << "|-- NODE"<<setw(2)<<i<<" PMM Read (MB/s):  "<<setw(8)<<md->iMC_PMM_Rd_socket[i]<<" --|";
        }
        cout << endl;
        for (uint32 i=skt; i<(skt+no_columns); ++i) {
            cout << "|-- NODE"<<setw(2)<<i<<" PMM Write(MB/s):  "<<setw(8)<<md->iMC_PMM_Wr_socket[i]<<" --|";
        }
        cout << endl;
        for (uint32 ctrl = 0; ctrl < max_imc_controllers; ++ctrl)
        {
            for (uint32 i=skt; i<(skt+no_columns); ++i) {
                cout << "|-- NODE"<<setw(2)<<i<<"."<<ctrl<<" NM read hit rate :"<<setw(6)<<md->M2M_NM_read_hit_rate[i][ctrl]<<" --|";
            }
            cout << endl;
        }
    }
    else
    {
        for (uint32 i=skt; i<(skt+no_columns); ++i) {
            cout << "|-- NODE"<<setw(2)<<i<<" P. Write (T/s): "<<dec<<setw(10)<<md->partial_write[i]<<" --|";
        }
        cout << endl;
    }
    for (uint32 i=skt; i<(skt+no_columns); ++i) {
        cout << "|-- NODE"<<setw(2)<<i<<" Memory (MB/s): "<<setw(11)<<std::right<<(md->iMC_Rd_socket[i]+md->iMC_Wr_socket[i]+
              md->iMC_PMM_Rd_socket[i]+md->iMC_PMM_Wr_socket[i])<<" --|";
    }
    cout << endl;
    for (uint32 i=skt; i<(no_columns+skt); ++i) {
        cout << "|---------------------------------------|";
    }
    cout << endl;
}

void display_bandwidth(PCM *m, memdata_t *md, uint32 no_columns, const bool show_channel_output)
{
    float sysReadDRAM = 0.0, sysWriteDRAM = 0.0, sysReadPMM = 0.0, sysWritePMM = 0.0;
    uint32 numSockets = m->getNumSockets();
    uint32 skt = 0;
    cout.setf(ios::fixed);
    cout.precision(2);

    while(skt < numSockets)
    {
        // Full row
        if ( (skt+no_columns) <= numSockets )
        {
            printSocketBWHeader (no_columns, skt, show_channel_output);
        if (show_channel_output)
                printSocketChannelBW(m, md, no_columns, skt);
            printSocketBWFooter (no_columns, skt, md);
            for (uint32 i=skt; i<(skt+no_columns); i++) {
                sysReadDRAM += md->iMC_Rd_socket[i];
                sysWriteDRAM += md->iMC_Wr_socket[i];
                sysReadPMM += md->iMC_PMM_Rd_socket[i];
                sysWritePMM += md->iMC_PMM_Wr_socket[i];
            }
            skt += no_columns;
        }
        else //Display one socket in this row
        {
            if (m->MCDRAMmemoryTrafficMetricsAvailable())
            {
                cout << "\
                    \r|---------------------------------------||---------------------------------------|\n\
                    \r|--                              Processor socket " << skt << "                            --|\n\
                    \r|---------------------------------------||---------------------------------------|\n\
                    \r|--       DDR4 Channel Monitoring     --||--      MCDRAM Channel Monitoring    --|\n\
                    \r|---------------------------------------||---------------------------------------|\n\
                    \r";
                uint32 max_channels = max_imc_channels <= max_edc_channels ? max_edc_channels : max_imc_channels;
                if (show_channel_output) {
       float iMC_Rd, iMC_Wr, EDC_Rd, EDC_Wr;
                   for(uint64 channel = 0; channel < max_channels; ++channel)
                   {
                    if (channel <= max_imc_channels) {
                        iMC_Rd = md->iMC_Rd_socket_chan[skt][channel];
                        iMC_Wr = md->iMC_Wr_socket_chan[skt][channel];
	    }
	    else
	    {
		iMC_Rd = -1.0;
		iMC_Wr = -1.0;
	    }
	    if (channel <= max_edc_channels) {
                        EDC_Rd = md->EDC_Rd_socket_chan[skt][channel];
                        EDC_Wr = md->EDC_Wr_socket_chan[skt][channel];
	    }
	    else
	    {
		EDC_Rd = -1.0;
		EDC_Rd = -1.0;
	    }

	    if (iMC_Rd >= 0.0 && iMC_Wr >= 0.0 && EDC_Rd >= 0.0 && EDC_Wr >= 0.0)
		cout << "|-- DDR4 Ch " << channel <<": Reads (MB/s):" << setw(9)  << iMC_Rd
		     << " --||-- EDC Ch " << channel <<": Reads (MB/s):" << setw(10)  << EDC_Rd
		     << " --|\n|--            Writes(MB/s):" << setw(9) << iMC_Wr
		     << " --||--           Writes(MB/s):" << setw(10)  << EDC_Wr
		     <<" --|\n";
	    else if ((iMC_Rd < 0.0 || iMC_Wr < 0.0) && EDC_Rd >= 0.0 && EDC_Wr >= 0.0)
		cout << "|--                                  "
		     << " --||-- EDC Ch " << channel <<": Reads (MB/s):" << setw(10)  << EDC_Rd
		     << " --|\n|--                                  "
		     << " --||--           Writes(MB/s):" << setw(10)  << EDC_Wr
		     <<" --|\n";

	    else if (iMC_Rd >= 0.0 && iMC_Wr >= 0.0 && (EDC_Rd < 0.0 || EDC_Wr < 0.0))
		cout << "|-- DDR4 Ch " << channel <<": Reads (MB/s):" << setw(9)  << iMC_Rd
		     << " --||--                                  "
		     << " --|\n|--            Writes(MB/s):" << setw(9) << iMC_Wr
		     << " --||--                                  "
		     <<" --|\n";
	    else
		continue;
       }
                }
                cout << "\
                    \r|-- DDR4 Mem Read  (MB/s):"<<setw(11)<<md->iMC_Rd_socket[skt]<<" --||-- MCDRAM Read (MB/s):"<<setw(14)<<md->EDC_Rd_socket[skt]<<" --|\n\
                    \r|-- DDR4 Mem Write (MB/s):"<<setw(11)<<md->iMC_Wr_socket[skt]<<" --||-- MCDRAM Write(MB/s):"<<setw(14)<<md->EDC_Wr_socket[skt]<<" --|\n\
                    \r|-- DDR4 Memory (MB/s)   :"<<setw(11)<<md->iMC_Rd_socket[skt]+md->iMC_Wr_socket[skt]<<" --||-- MCDRAM (MB/s)     :"<<setw(14)<<md->EDC_Rd_socket[skt]+md->EDC_Wr_socket[skt]<<" --|\n\
                    \r|---------------------------------------||---------------------------------------|\n\
                    \r";

                sysReadDRAM  += (md->iMC_Rd_socket[skt]+md->EDC_Rd_socket[skt]);
                sysWriteDRAM += (md->iMC_Wr_socket[skt]+md->EDC_Wr_socket[skt]);
                skt += 1;
            }
        else
        {
                cout << "\
                    \r|---------------------------------------|\n\
                    \r|--             Socket "<<skt<<"              --|\n\
                    \r|---------------------------------------|\n";
                if (show_channel_output) {
      cout << "\
                    \r|--     Memory Channel Monitoring     --|\n\
                    \r|---------------------------------------|\n\
                    \r"; 
                  for(uint64 channel = 0; channel < max_imc_channels; ++channel)
                  {
                    if(md->iMC_Rd_socket_chan[skt][channel] < 0.0 && md->iMC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
                        continue;
                    cout << "|--  Mem Ch " << channel <<": Reads (MB/s):" << setw(8)  << md->iMC_Rd_socket_chan[skt][channel]
                        <<"  --|\n|--            Writes(MB/s):" << setw(8) << md->iMC_Wr_socket_chan[skt][channel]
                        <<"  --|\n";
                    if (md->PMM)
                    {
                        cout << "|--      PMM Reads (MB/s):" << setw(8) << md->iMC_PMM_Rd_socket_chan[skt][channel] << "  --|\n";
                        cout << "|--      PMM Writes(MB/s):" << setw(8) << md->iMC_PMM_Wr_socket_chan[skt][channel] << "  --|\n";
                    }
                  }
    }
                cout << "\
                    \r|-- NODE"<<skt<<" Mem Read (MB/s)  :"<<setw(8)<<md->iMC_Rd_socket[skt]<<"  --|\n\
                    \r|-- NODE"<<skt<<" Mem Write (MB/s) :"<<setw(8)<<md->iMC_Wr_socket[skt]<<"  --|\n";
                if(md->PMM)
                {
                    cout << "\
                        \r|-- NODE"<<skt<<" PMM Read (MB/s):"<<setw(8)<<md->iMC_PMM_Rd_socket[skt]<<"  --|\n\
                        \r|-- NODE"<<skt<<" PMM Write(MB/s):"<<setw(8)<<md->iMC_PMM_Wr_socket[skt]<<"  --|\n";
                    for (uint32 ctrl = 0; ctrl < max_imc_controllers; ++ctrl)
                    {
                        cout << "\r|-- NODE"<<setw(2)<<skt<<"."<<ctrl<<" NM read hit rate :"<<setw(6)<<md->M2M_NM_read_hit_rate[skt][ctrl]<<" --|\n";
                    }
                }
                else
                {
                    cout <<
                       "\r|-- NODE"<<skt<<" P. Write (T/s) :"<<setw(10)<<dec<<md->partial_write[skt]<<"  --|\n";
                }
                cout <<
                   "\r|-- NODE"<<skt<<" Memory (MB/s): "<<setw(8)<<md->iMC_Rd_socket[skt]+md->iMC_Wr_socket[skt]+
                    md->iMC_PMM_Rd_socket[skt]+md->iMC_PMM_Wr_socket[skt]<<"     --|\n\
                    \r|---------------------------------------|\n\
                    \r";

                sysReadDRAM += md->iMC_Rd_socket[skt];
                sysWriteDRAM += md->iMC_Wr_socket[skt];
                sysReadPMM += md->iMC_PMM_Rd_socket[skt];
                sysWritePMM += md->iMC_PMM_Wr_socket[skt];
                skt += 1;
            }
        }
    }
    {
        cout << "\
            \r|---------------------------------------||---------------------------------------|\n";
    if(md->PMM)
           cout << "\
            \r|--            System DRAM Read Throughput(MB/s):"<<setw(14)<<sysReadDRAM<<"                --|\n\
            \r|--           System DRAM Write Throughput(MB/s):"<<setw(14)<<sysWriteDRAM<<"                --|\n\
            \r|--             System PMM Read Throughput(MB/s):"<<setw(14)<<sysReadPMM<<"                --|\n\
            \r|--            System PMM Write Throughput(MB/s):"<<setw(14)<<sysWritePMM<<"                --|\n";
        cout << "\
            \r|--                 System Read Throughput(MB/s):"<<setw(14)<<sysReadDRAM+sysReadPMM<<"                --|\n\
            \r|--                System Write Throughput(MB/s):"<<setw(14)<<sysWriteDRAM+sysWritePMM<<"                --|\n\
            \r|--               System Memory Throughput(MB/s):"<<setw(14)<<sysReadDRAM+sysReadPMM+sysWriteDRAM+sysWritePMM<<"                --|\n\
            \r|---------------------------------------||---------------------------------------|" << endl;
    }
}

void display_bandwidth_csv_header(PCM *m, memdata_t *md, const bool show_channel_output)
{
    uint32 numSockets = m->getNumSockets();
    cout << ",," ; // Time

    for (uint32 skt=0; skt < numSockets; ++skt)
    {
      if (show_channel_output) {
         for(uint64 channel = 0; channel < max_imc_channels; ++channel)
         {
         if(md->iMC_Rd_socket_chan[skt][channel] < 0.0 && md->iMC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
            continue;
         cout << "SKT" << skt << ",SKT" << skt << ',';
             if (md->PMM)
             {
                 cout << "SKT" << skt << ",SKT" << skt << ',';
             }
         }
      }
      cout << "SKT"<<skt<<","
       << "SKT"<<skt<<","
       << "SKT"<<skt<<",";
      if (m->getCPUModel() != PCM::KNL) {
          if (md->PMM)
          cout << "SKT"<<skt<<"," << "SKT"<<skt<<",";
          else
              cout << "SKT"<<skt<<",";
      }

      if (m->MCDRAMmemoryTrafficMetricsAvailable()) {
      if (show_channel_output) {
             for(uint64 channel = 0; channel < max_edc_channels; ++channel)
             {
             if(md->EDC_Rd_socket_chan[skt][channel] < 0.0 && md->EDC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
                 continue;
             cout << "SKT" << skt << ",SKT" << skt << ',';
         }
      }
          cout << "SKT"<<skt<<","
               << "SKT"<<skt<<","
           << "SKT"<<skt<<",";
      }

    }
    if (md->PMM)
        cout << "System,System,System,System,";
    cout << "System,System,System\n";

    cout << "Date,Time," ;
    for (uint32 skt=0; skt < numSockets; ++skt)
    {
      if (show_channel_output) {
         for(uint64 channel = 0; channel < max_imc_channels; ++channel)
         {
         if(md->iMC_Rd_socket_chan[skt][channel] < 0.0 && md->iMC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
             continue;
         cout << "Ch" <<channel <<"Read,"
              << "Ch" <<channel <<"Write,";
             if(md->PMM)
             {
                 cout << "Ch" <<channel <<"PMM_Read,"
                      << "Ch" <<channel <<"PMM_Write,";
             }
     }
      }
      if (m->getCPUModel() == PCM::KNL)
          cout << "DDR4 Read (MB/s), DDR4 Write (MB/s), DDR4 Memory (MB/s),";
      else
      {
          if(md->PMM)
              cout << "Mem Read (MB/s),Mem Write (MB/s), PMM_Read, PMM_Write; Memory (MB/s),";
          else
              cout << "Mem Read (MB/s),Mem Write (MB/s), P. Write (T/s), Memory (MB/s),";
      }

      if (m->MCDRAMmemoryTrafficMetricsAvailable()) {
         if (show_channel_output) {
            for(uint64 channel = 0; channel < max_edc_channels; ++channel)
            {
             if(md->EDC_Rd_socket_chan[skt][channel] < 0.0 && md->EDC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
                 continue;
             cout << "EDC_Ch" <<channel <<"Read,"
                  << "EDC_Ch" <<channel <<"Write,";
           }
     }
         cout << "MCDRAM Read (MB/s), MCDRAM Write (MB/s), MCDRAM (MB/s),";
      }
    }

    if (md->PMM)
    cout << "DRAMRead,DRAMWrite,PMMREAD;PMMWrite,";
    cout << "Read,Write,Memory" << endl;
}

void display_bandwidth_csv(PCM *m, memdata_t *md, uint64 elapsedTime, const bool show_channel_output)
{
    uint32 numSockets = m->getNumSockets();
    tm tt = pcm_localtime();
    cout.precision(3);
    cout << 1900+tt.tm_year << '-' << 1+tt.tm_mon << '-' << tt.tm_mday << ','
         << tt.tm_hour << ':' << tt.tm_min << ':' << tt.tm_sec << ',';


    float sysReadDRAM = 0.0, sysWriteDRAM = 0.0, sysReadPMM = 0.0, sysWritePMM = 0.0;

    cout.setf(ios::fixed);
    cout.precision(2);

    for (uint32 skt=0; skt < numSockets; ++skt)
    {
    if (show_channel_output) {
           for(uint64 channel = 0; channel < max_imc_channels; ++channel)
           {
          if(md->iMC_Rd_socket_chan[skt][channel] < 0.0 && md->iMC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
             continue;
          cout <<setw(8) << md->iMC_Rd_socket_chan[skt][channel] << ','
               <<setw(8) << md->iMC_Wr_socket_chan[skt][channel] << ',';
              if(md->PMM)
              {
                  cout <<setw(8) << md->iMC_PMM_Rd_socket_chan[skt][channel] << ','
                       <<setw(8) << md->iMC_PMM_Wr_socket_chan[skt][channel] << ',';
              }
       }
     }
         cout <<setw(8) << md->iMC_Rd_socket[skt] <<','
          <<setw(8) << md->iMC_Wr_socket[skt] <<',';
         if(md->PMM)
         {
             cout <<setw(8) << md->iMC_PMM_Rd_socket[skt] <<','
                  <<setw(8) << md->iMC_PMM_Wr_socket[skt] <<',';
         }
     if (m->getCPUModel() != PCM::KNL)
         {
             if (!md->PMM)
             {
                 cout <<setw(10) << dec << md->partial_write[skt] <<',';
             }
         }
         cout << setw(8) << md->iMC_Rd_socket[skt]+md->iMC_Wr_socket[skt] <<',';

     sysReadDRAM += md->iMC_Rd_socket[skt];
         sysWriteDRAM += md->iMC_Wr_socket[skt];
         sysReadPMM += md->iMC_PMM_Rd_socket[skt];
         sysWritePMM += md->iMC_PMM_Wr_socket[skt];

     if (m->MCDRAMmemoryTrafficMetricsAvailable()) {
            if (show_channel_output) {
           for(uint64 channel = 0; channel < max_edc_channels; ++channel)
           {
                  if(md->EDC_Rd_socket_chan[skt][channel] < 0.0 && md->EDC_Wr_socket_chan[skt][channel] < 0.0) //If the channel read neg. value, the channel is not working; skip it.
                   continue;
                  cout <<setw(8) << md->EDC_Rd_socket_chan[skt][channel] << ','
                   <<setw(8) << md->EDC_Wr_socket_chan[skt][channel] << ',';

           }
        }
             cout <<setw(8) << md->EDC_Rd_socket[skt] <<','
              <<setw(8) << md->EDC_Wr_socket[skt] <<','
                  <<setw(8) << md->EDC_Rd_socket[skt]+md->EDC_Wr_socket[skt] <<',';

             sysReadDRAM += md->EDC_Rd_socket[skt];
             sysWriteDRAM += md->EDC_Wr_socket[skt];
     }
    }

    if (md->PMM)
        cout <<setw(10) <<sysReadDRAM <<','
             <<setw(10) <<sysWriteDRAM <<','
             <<setw(10) <<sysReadPMM <<','
             <<setw(10) <<sysWritePMM <<',';

    cout <<setw(10) <<sysReadDRAM+sysReadPMM <<','
     <<setw(10) <<sysWriteDRAM+sysWritePMM <<','
     <<setw(10) <<sysReadDRAM+sysReadPMM+sysWriteDRAM+sysWritePMM << endl;
}

void calculate_bandwidth(PCM *m, const ServerUncorePowerState uncState1[], const ServerUncorePowerState uncState2[], uint64 elapsedTime, bool csv, bool & csvheader, uint32 no_columns, bool PMM, const bool show_channel_output)
{
    //const uint32 num_imc_channels = m->getMCChannelsPerSocket();
    //const uint32 num_edc_channels = m->getEDCChannelsPerSocket();
    memdata_t md;
    md.PMM = PMM;

    for(uint32 skt = 0; skt < m->getNumSockets(); ++skt)
    {
        md.iMC_Rd_socket[skt] = 0.0;
        md.iMC_Wr_socket[skt] = 0.0;
        md.iMC_PMM_Rd_socket[skt] = 0.0;
        md.iMC_PMM_Wr_socket[skt] = 0.0;
        md.EDC_Rd_socket[skt] = 0.0;
        md.EDC_Wr_socket[skt] = 0.0;
        md.partial_write[skt] = 0;
        for(uint32 i=0; i < max_imc_controllers; ++i)
        {
            md.M2M_NM_read_hit_rate[skt][i] = 0.;
        }
        const uint32 numChannels1 = m->getMCChannels(skt, 0); // number of channels in the first controller

    switch(m->getCPUModel()) {
    case PCM::KNL:
            for(uint32 channel = 0; channel < max_edc_channels; ++channel)
            {
                if(getEDCCounter(channel,READ,uncState1[skt],uncState2[skt]) == 0.0 && getEDCCounter(channel,WRITE,uncState1[skt],uncState2[skt]) == 0.0)
                {
                    md.EDC_Rd_socket_chan[skt][channel] = -1.0;
                    md.EDC_Wr_socket_chan[skt][channel] = -1.0;
                    continue;
                }

                md.EDC_Rd_socket_chan[skt][channel] = (float) (getEDCCounter(channel,READ,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0));
                md.EDC_Wr_socket_chan[skt][channel] = (float) (getEDCCounter(channel,WRITE,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0));

                md.EDC_Rd_socket[skt] += md.EDC_Rd_socket_chan[skt][channel];
                md.EDC_Wr_socket[skt] += md.EDC_Wr_socket_chan[skt][channel];
        }
        break;
        default:
            for(uint32 channel = 0; channel < max_imc_channels; ++channel)
            {
                if(getMCCounter(channel,READ,uncState1[skt],uncState2[skt]) == 0.0 && getMCCounter(channel,WRITE,uncState1[skt],uncState2[skt]) == 0.0) //In case of JKT-EN, there are only three channels. Skip one and continue.
                {
                    if (!PMM || (getMCCounter(channel,PMM_READ,uncState1[skt],uncState2[skt]) == 0.0 && getMCCounter(channel,PMM_WRITE,uncState1[skt],uncState2[skt]) == 0.0))
                    {
                        md.iMC_Rd_socket_chan[skt][channel] = -1.0;
                        md.iMC_Wr_socket_chan[skt][channel] = -1.0;
                        continue;
                    }
                }

                md.iMC_Rd_socket_chan[skt][channel] = (float) (getMCCounter(channel,READ,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0));
                md.iMC_Wr_socket_chan[skt][channel] = (float) (getMCCounter(channel,WRITE,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0));

                md.iMC_Rd_socket[skt] += md.iMC_Rd_socket_chan[skt][channel];
                md.iMC_Wr_socket[skt] += md.iMC_Wr_socket_chan[skt][channel];

                if(PMM)
                {
                    md.iMC_PMM_Rd_socket_chan[skt][channel] = (float) (getMCCounter(channel,PMM_READ,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0));
                    md.iMC_PMM_Wr_socket_chan[skt][channel] = (float) (getMCCounter(channel,PMM_WRITE,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0));

                    md.iMC_PMM_Rd_socket[skt] += md.iMC_PMM_Rd_socket_chan[skt][channel];
                    md.iMC_PMM_Wr_socket[skt] += md.iMC_PMM_Wr_socket_chan[skt][channel];

                    md.M2M_NM_read_hit_rate[skt][(channel < numChannels1)?0:1] += (float)getMCCounter(channel,READ,uncState1[skt],uncState2[skt]);
                }
                else
                {
                    md.partial_write[skt] += (uint64) (getMCCounter(channel,PARTIAL,uncState1[skt],uncState2[skt]) / (elapsedTime/1000.0));
                }
            }
    }
        if (PMM)
        {
            for(uint32 c = 0; c < max_imc_controllers; ++c)
            {
                if(md.M2M_NM_read_hit_rate[skt][c] != 0.0)
                {
                    md.M2M_NM_read_hit_rate[skt][c] = ((float)getM2MCounter(c, NM_HIT, uncState1[skt],uncState2[skt]))/ md.M2M_NM_read_hit_rate[skt][c];
                }
            }
        }
    }

    if (csv) {
      if (csvheader) {
    display_bandwidth_csv_header(m, &md, show_channel_output);
    csvheader = false;
      }
      display_bandwidth_csv(m, &md, elapsedTime, show_channel_output);
    } else {
      display_bandwidth(m, &md, no_columns, show_channel_output);
    }
}

void calculate_bandwidth(PCM *m, const ServerUncorePowerState uncState1[], const ServerUncorePowerState uncState2[], uint64 elapsedTime, bool csv, bool & csvheader, uint32 no_columns, int rankA, int rankB)
{
    uint32 skt = 0;
    cout.setf(ios::fixed);
    cout.precision(2);
    uint32 numSockets = m->getNumSockets();

    while(skt < numSockets)
    {
        // Full row
        if ( (skt+no_columns) <= numSockets )
        {
            printSocketRankBWHeader(no_columns, skt);
            printSocketChannelBW(no_columns, skt, max_imc_channels, uncState1, uncState2, elapsedTime, rankA, rankB);
            for (uint32 i=skt; i<(no_columns+skt); ++i) {
              cout << "|-------------------------------------------|";
            }
            cout << endl;
            skt += no_columns;
        }
        else //Display one socket in this row
        {
            cout << "\
                \r|-------------------------------------------|\n\
                \r|--               Socket "<<skt<<"                --|\n\
                \r|-------------------------------------------|\n\
                \r|--           DIMM Rank Monitoring        --|\n\
                \r|-------------------------------------------|\n\
                \r";
            for(uint32 channel = 0; channel < max_imc_channels; ++channel)
            {
                if(rankA >=0)
                  cout << "|-- Mem Ch "
                      << setw(2) << channel
                      << " R " << setw(1) << rankA
                      <<": Reads (MB/s):"
                      <<setw(8)
                      <<(float) (getMCCounter(channel,READ_RANK_A,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0))
                      <<"  --|\n|--                Writes(MB/s):"
                      <<setw(8)
                      <<(float) (getMCCounter(channel,WRITE_RANK_A,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0))
                      <<"  --|\n";
                if(rankB >=0)
                  cout << "|-- Mem Ch "
                      << setw(2) << channel
                      << " R " << setw(1) << rankB
                      <<": Reads (MB/s):"
                      <<setw(8)
                      <<(float) (getMCCounter(channel,READ_RANK_B,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0))
                      <<"  --|\n|--                Writes(MB/s):"
                      <<setw(8)
                      <<(float) (getMCCounter(channel,WRITE_RANK_B,uncState1[skt],uncState2[skt]) * 64 / 1000000.0 / (elapsedTime/1000.0))
                      <<"  --|\n";
            }
            cout << "\
                \r|-------------------------------------------|\n\
                \r";

            skt += 1;
        }
    }
}
