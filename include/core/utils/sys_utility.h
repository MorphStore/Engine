/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   sys_utility.h
 * Author: Annett
 *
 * Created on 23. Juli 2019, 14:39
 */

#ifndef SYS_UTILITY_H
#define SYS_UTILITY_H

#include <stdio.h>
#include <dirent.h>
#include <vector>

namespace morphstore {
    
class EnergyContainer{
public:
    std::vector<long long> E_start;
    std::vector<long long> E_end;
    std::vector<std::string> E_files; 

    void startMeasurement(){
        for (size_t i=0; i< E_files.size(); i++){
            file=fopen(E_files[i].c_str(),"r");
            fscanf(file,"%lld",&E_start[i]);
            fclose(file);
        }
    }
    
    void endMeasurement(){
        for (size_t i=0; i< E_files.size(); i++){
            file=fopen(E_files[i].c_str(),"r");
            fscanf(file,"%lld",&E_end[i]);
            fclose(file);
        }
    }
    
    EnergyContainer(){
        
        struct dirent *entry;
        DIR *dp;
        #ifdef RAPL
        dp = opendir("/sys/class/powercap/intel-rapl/");
   
        
        if (dp == NULL) 
        {
          perror("opendir");
          
        }
        
        while((entry = readdir(dp))) 
      
            if (12 <= strlen(entry->d_name) && ( strncmp("intel-rapl:",entry->d_name,11) == 0 ))
        
            {
                E_files.push_back("/sys/class/powercap/intel-rapl/" + std::string(entry->d_name) + "/energy_uj");
                E_start.push_back(0);
                E_end.push_back(0);
            }
        
        closedir(dp);
        #endif 
    }
    
private:
    FILE *file;
};

}
#endif /* SYS_UTILITY_H */

