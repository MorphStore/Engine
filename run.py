from filecmp import dircmp
import subprocess
import os
import shutil
import sys
import stat
import argparse
import re

def run_command(cmd, logfile_str, errfile_str):
    logfile = open(logfile_str, "w")
    errfile = open(errfile_str, "w")
    p = subprocess.Popen(
        cmd,
        # shell=True,
        universal_newlines=True,
        stdout=logfile,
        stderr=subprocess.PIPE
    )
    for line in p.stderr:
        sys.stderr.write(line)
        errfile.write(line)
    ret_code = p.wait()
    logfile.flush()
    errfile.flush()
    logfile.close()
    errfile.close()
    return ret_code

llvm_base_dir="/opt/nec/nosupport"
llvm_version="llvm-ve-1.7.0"

dirpath = os.getcwd()




cmake_compiler_flags=[]
cmake_compiler_flags.append("-DCMAKE_BUILD_TYPE=Debug")
cmake_compiler_flags.append("-DNO_SELF_MANAGING=True")
cmake_compiler_flags.append("-DCOMPILER_ID=CLANG")
cmake_compiler_flags.append("-DCTSUBASA=True")
compiler_dir="{}/{}".format(llvm_base_dir, llvm_version)
cmake_clang_search_path="{}/lib/cmake/clang".format(compiler_dir)
cmake_llvm_search_path="{}/lib/cmake/llvm".format(compiler_dir)
cmake_compiler_flags.append("-DCMAKE_C_COMPILER={}/bin/clang".format(compiler_dir))
cmake_compiler_flags.append("-DCMAKE_CXX_COMPILER={}/bin/clang++".format(compiler_dir))
cmake_compiler_flags.append("-DCMAKE_CLANG_SEARCH_PATH={}".format(cmake_clang_search_path))
cmake_compiler_flags.append("-DCMAKE_LLVM_SEARCH_PATH={}".format(cmake_llvm_search_path))
platform="SXAURORA"
cmake_misc_flags = ["-DCPLATFORM={}".format(platform), "--debug-trycompile"]
cmake_command = ["cmake"]
cmake_command.extend(cmake_compiler_flags)
cmake_command.extend(cmake_misc_flags)
cmake_command.append("..")

if os.path.exists(dirpath + "/build"):
    shutil.rmtree(dirpath + "/build")
os.makedirs(dirpath + "/build")
os.makedirs(dirpath + "/build/log")

os.chdir(dirpath + "/build")

print("Executing {}".format(cmake_command))
cmake_return = run_command(
    cmake_command,
    "log/cmake.log",
    "log/cmake.err"
)
if cmake_return != 0:
    print("Error in cmake. Look into " + dirpath + "/build/log/cmake.[err|log]")
    sys.exit()
print("Success.")
print("Executing  MAKE... ", end='', flush=True)
make_return = run_command(
    [
        "make",
        "-j",
        "VERBOSE=True"
    ],
    "log/make.log",
    "log/make.err"
)
if make_return != 0:
    print("Error in make. Look into " + dirpath + "/build/log/make.[err|log]")
    sys.exit()
else:
    print("Sucess.")
