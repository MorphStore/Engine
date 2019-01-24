#!/usr/bin/env bash
rm -rf build/
cd build/
cmake ..
make -j4
