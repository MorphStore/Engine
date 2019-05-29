/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   my_operator.h
 * Author: Annett
 *
 * Created on 30. April 2019, 16:06
 */

#ifndef MY_OPERATOR_H
#define MY_OPERATOR_H

#include <vector/general_vector.h>
#include <vector/primitives/calc.h>
#include <vector/primitives/create.h>


template<class VectorExtension>
      
      int my_operator(int number) {
      
        using namespace vector;

        IMPORT_VECTOR_BOILER_PLATE(VectorExtension)
        
        vector_t vec = set1<VectorExtension, vector_base_t_granularity::value>(number);
        return hadd<VectorExtension, vector_base_t_granularity::value>::apply(vec);
        
      }

#endif /* MY_OPERATOR_H */

