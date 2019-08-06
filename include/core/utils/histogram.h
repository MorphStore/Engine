/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   histogram.h
 * Author: Annett
 *
 * Created on 27. Juni 2019, 21:40
 */

#ifndef HISTOGRAM_H
#define HISTOGRAM_H

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

namespace morphstore{
    
    template<class t_src_f>
    unsigned* get_histogram(const column<t_src_f> * inCol){
        
        unsigned * histogram = new unsigned[64]();
                
        auto uncompr_column = safe_morph<uncompr_f,t_src_f>(inCol);
        
        for (size_t i=0; i < uncompr_column->get_count_values(); i++){
            histogram[effective_bitwidth(((uint64_t *) uncompr_column->get_data())[i])-1]++;
        }
        
        return histogram;
    }
}

#endif /* HISTOGRAM_H */

