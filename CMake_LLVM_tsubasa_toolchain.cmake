# clone frome
# takes NMPI_ROOT and NLC_ROOT from environment
#set(CMAKE_SYSTEM_NAME Aurora-VE)

#todo: make relative pathes with base directory as parameter from build.sh

set(LLVM_PATH /opt/nec/nosupport/llvm-ve)

set(CMAKE_CXX_COMPILER ${LLVM_PATH}/bin/clang++ CACHE FILEPATH "Aurora LLVM C++ compiler")
set(CMAKE_C_COMPILER ${LLVM_PATH}/bin/clang CACHE FILEPATH "Aurora LLVM C compiler")
set(CMAKE_AR ${LLVM_PATH}/bin/llvm-ar CACHE FILEPATH "Aurora LLVM archiver")
set(CMAKE_RANLIB ${LLVM_PATH}/bin/ranlib CACHE FILEPATH "Aurora LLVM ranlib")
set(CMAKE_C_COMPILER_TARGET "ve-linux")
set(CMAKE_CXX_COMPILER_TARGET "ve-linux")
set(LLVM_CONFIG_PATH ${LLVM_PATH}/bin/llvm-config CACHE FILEPATH "Aurora LLVM Config")