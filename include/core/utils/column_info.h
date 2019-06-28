/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   column_info.h
 * Author: Annett
 *
 * Created on 28. Juni 2019, 15:31
 */

#ifndef COLUMN_INFO_H
#define COLUMN_INFO_H

#include <core/memory/mm_glob.h>
#include <core/storage/column.h>

#include <core/utils/basic_types.h>
#include <core/utils/preprocessor.h>
#include <core/utils/math.h>

#include <core/morphing/format.h>
#include <core/morphing/safe_morph.h>
#include <core/morphing/uncompr.h>
#include <core/morphing/static_vbp.h>
#include <core/morphing/dynamic_vbp.h>
#include <limits>

#include <cstdint>

//Get Information about columns. Note: This is terribly slow, don't use it for benchmarks!

namespace morphstore{
    
    template<class t_src_f>
    int get_sorted(const column<t_src_f> * inCol){
        
        
        auto uncompr_column = safe_morph<uncompr_f,t_src_f>(inCol);
        uint64_t * data = (uint64_t *) (uncompr_column->get_data());
        if (uncompr_column->get_count_values() == 1) return 1;
        
        unsigned up=0;
        unsigned new_up=0;
        
        if (data[0]<data[1]) up=1;
        
        for (size_t i=1; i < uncompr_column->get_count_values(); i++){
            if (data[i-1]<data[i]) new_up=1;
            if (data[i-1]>data[i]) new_up=0;
            //no change when values are equal
            if (up!=new_up) return 0;
        }
        
        return 1;
    }
    
    template<class t_src_f>
    int get_unique(const column<t_src_f> * inCol){
        
        auto uncompr_column = safe_morph<uncompr_f,t_src_f>(inCol);
        uint64_t * data = (uint64_t *) (uncompr_column->get_data());
        
        if (uncompr_column->get_count_values() == 1) return 1;
        
        for (size_t i=0; i < uncompr_column->get_count_values(); i++){
            
                for (size_t j=0; j < uncompr_column->get_count_values(); j++){
                    if ((data[j]==data[i]) && (i != j)) return 0;
                }
        }
        
        return 1;
    }
    
}

#endif /* COLUMN_INFO_H */

