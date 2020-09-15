#!/bin/bash

numactl --membind=0 --cpunodebind=0 ./build/release/src/parallelization/virtual_vector_query
