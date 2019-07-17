# clone frome
# takes NMPI_ROOT and NLC_ROOT from environment
#set(CMAKE_SYSTEM_NAME Aurora-VE)

#todo: make relative pathes with base directory as parameter from build.sh

set(LLVM_PATH /opt/nec/nosupport/llvm-ve)
#link_directories(${LLVM_PATH}/lib)
#include_directories(${LLVM_PATH}/include)

set(CMAKE_CXX_COMPILER /opt/nec/nosupport/llvm-ve/bin/clang++ CACHE FILEPATH "Aurora LLVM C++ compiler")
set(CMAKE_C_COMPILER /opt/nec/nosupport/llvm-ve/bin/clang CACHE FILEPATH "Aurora LLVM C compiler")

#unset(CMAKE_LINKER CACHE)
#set(CMAKE_LINKER /opt/nec/ve/LLVM/llvm-ve-rv-1.1.2/bin/llvm-link CACHE FILEPATH "Aurora LLVM linker" FORCE)
#
#unset(CMAKE_CXX_LINKER CACHE)
set(CMAKE_CXX_LINKER /opt/nec/nosupport/llvm-ve/bin/llvm-link)
set(CMAKE_C_LINKER /opt/nec/nosupport/llvm-ve/bin/llvm-link)
set(CMAKE_AR /opt/nec/nosupport/llvm-ve/bin/llvm-ar CACHE FILEPATH "Aurora LLVM archiver")
set(CMAKE_NM /opt/nec/nosupport/llvm-ve/bin/llvm-nm CACHE FILEPATH "Aurora LLVM nm")
set(CMAKE_OBJDUMP /opt/nec/nosupport/llvm-ve/bin/llvm-objdump CACHE FILEPATH "Aurora LLVM objdump")
set(CMAKE_RANLIB /opt/nec/nosupport/llvm-ve/bin/ranlib CACHE FILEPATH "Aurora LLVM ranlib")



#set(CMAKE_C_FLAGS   "-U__GNUC__ -U__GNUC_MINOR__" CACHE STRING "" FORCE)
#set(CMAKE_CXX_FLAGS "-U__GNUC__ -U__GNUC_MINOR__" CACHE STRING "" FORCE)

#set(CMAKE_CROSSCOMPILING_EMULATOR "/opt/nec/ve/bin/ve_exec" CACHE FILEPATH "Command to execute VE binaries")