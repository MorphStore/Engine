/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>

#ifdef AVXTWO
#include <vector/simd/avx2/extension_avx2.h>
#include <vector/simd/avx2/primitives/calc_avx2.h>
#include <vector/simd/avx2/primitives/create_avx2.h>
#endif

#include <vector/simd/sse/extension_sse.h>
#include <vector/simd/sse/primitives/calc_sse.h>
#include <vector/simd/sse/primitives/create_sse.h>
#include "my_operator.h"

int main(){  
  
  using namespace vector;
  int result = 0;
  
  #ifdef AVXTWO 
  result = my_operator<avx2<v256<uint64_t>>>( 1 );
  std::cout << "Result sum (avx2): " << result << std::endl;
  #endif
  
   
  result = my_operator<sse<v128<uint64_t>>>( 1 );
  std::cout << "Result sum (sse): " << result << std::endl;
  
  
  return 0;
}