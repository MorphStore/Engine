#include "../../include/core/memory/mm_glob.h"
#include "../../include/core/morphing/format.h"

#include "../../include/core/utils/basic_types.h"


#include <functional>
#include <iostream>
#include <random>

#include <fstream>
#include <vector>

#include <math.h>

#include <chrono>

using namespace morphstore;
using namespace vectorlib;

static inline uint64_t now();
static inline double time_elapsed_ns( uint64_t start, uint64_t end );

static inline uint64_t now() {
   uint64_t  ret;
   asm volatile( "smir %0, %%usrcc":"=r"(ret) );
   return ret;
}
static inline double time_elapsed_ns( uint64_t start, uint64_t end ) {
   return ( (double)( end - start ) ) / 1.4;
}