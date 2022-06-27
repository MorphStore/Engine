#!/bin/bash
function is_power_of_two () {
    declare -i n=$1
    (( n > 0 && (n & (n - 1)) == 0 ))
}

function is_int () {
    re="^[0-9]+$"
    if [[ $1 =~ $re ]]
    then
        return 0
    else
        return 1
    fi
}

function printHelp {
    echo "build.sh -buildMode [-loggerControl] [-memory] [-jN]"
    echo "buildMode:"
    echo "    -hi|--hiPerformance"
    echo "         Release mode, but with O3 and link time optimization"
    echo "    -rel|--release"
    echo "         Release mode"
    echo "    -deb|--debug"
    echo "         Debug mode"
    echo ""
    echo "loggerControl:"
    echo "    -nl|--noLog"
    echo "         Completely removes all calls to the logger."
    echo ""
    echo "memory:"
    echo "    -debM"
    echo "         Prints calls to the Memory manager on the debug channel."
    echo "    -noSelfManaging"
    echo "         Deactivates the memory hooks and enables standard C++ memory management."
    echo "    -queryMinMemEx 2^x"
    echo "         Sets the minimum chunk reallocation size in bytes for the Query Memory Manager. The value should be a power of two."
    echo "    -queryInitSize x"
    echo "         Sets the initial amount of memory in bytes allocated by the Query Memory Manager."
    echo "    -queryDisallowExpand"
    echo "         Disallows the Query Memory Manager to allocate additional memory beyond its initial amount. Use with care: set -queryInitSize to a sufficient amount."
    echo "    -queryInitBuffers"
    echo "         Make the Query Memory Manager initialize all buffers it allocates to zero directly after their allocation."
    echo "    -lc|--leakCheck"
    echo "         Makes the MemoryManager more aware of possible memory leaks."
    echo "    --alignment 2^x"
    echo "         Sets the byte alignment for the MemoryManager"
    echo ""
    echo "    -mon|--enable-monitoring"
    echo "         Embedds the monitoring macros which are optimized out otherwise."
    echo ""
    echo "    -jN:"
    echo "         N > 0 sets the number of parallel make jobs"
    echo ""
    echo "sanity:"
    echo "    -tA|--testAll"
    echo "         Runs CTest for all layers"
    echo "    -tMm|--testMem"
    echo "         Runs CTest for the memory manager"
    echo "    -tMo|--testMorph"
    echo "         Runs CTest for the morphing layer"
    echo "    -tOp|--testOps"
    echo "         Runs CTest for the operators"
    echo "    -tPe|--testPers"
    echo "         Runs CTest for the persistence layer"
    echo "    -tSt|--testStorage"
    echo "         Runs CTest for the storage layer"
    echo "    -tUt|--testUtils"
    echo "         Runs CTest for some utilities"
    echo "    -tVt|--testVectoring"
    echo "         Runs CTest for vectorized "
        echo ""
        echo "targets:"
        echo "    -bA|--buildAll"
        echo "         Builds all targets in the src directory"
        echo "    -bCa|--buildCalibration"
        echo "         Builds all calibration micro-benchmarks"
        echo "    -bEx|--buildExamples"
        echo "         Builds all examples"
        echo "    -bMbm|--buildMicroBms"
        echo "         Builds all micro-benchmarks"
        echo "    -bSSB sf|--buildSSB sf"
        echo "         Builds all SSB queries for scale factor sf (if the generated sources are available)"
        echo "    --target TARGETNAME"
        echo "         Builds only the target TARGETNAME, which must be included in one of the target groups selected using the above \"-b\"-arguments"
        echo "         It is possible to specify multiple target names by providing a quoted white-space-separated list for TARGETNAME"
        echo "         Defaults to \"all\", i.e., if omited, all targets of the selected target groups will be built"
    echo ""
  echo "features:"
    echo "    -avx512"
    echo "         Builds with avx512 and avx2 support"
    echo "    -avxtwo"
    echo "         Builds with avx2 support"
  echo "    -sse4"
    echo "         Builds with sse4.2 support"
  echo "    -armneon"
    echo "         Builds with neon support (for ARM)"
  echo "    -armsve"
    echo "         Builds with SVE support (for ARM, work in progress)"
  echo "  --tvl PATH"
  echo "       Provide an alternative path for the TVL"
        echo ""
        echo "energy:"
        echo "    -rapl"
        echo "         Builds with RAPL support"
        echo "    -odroid"
        echo "         Builds with support for the Odroid-XU3 sensors"
        echo ""
        echo "compression:"
        echo "    --vbpLimitRoutinesForSSBSF1"
        echo "         Build the vertical bit-packing routines only for the bit widths required for executing SSB at scale factor 1, to speed up the build."
        echo "       These are also sufficient for scale factor 10"
        echo "    --vbpLimitRoutinesForSSBSF100"
        echo "         Build the vertical bit-packing routines only for the bit widths required for executing SSB at scale factor 100, to speed up the build."
        echo "misc:"
        echo "    --projectAssumePrepared"
        echo "         The project operator on compressed data will not check whether its input data column was prepared for random access."
}

buildType=""
makeParallel="1"

buildModeSet=0

logging="-UNOLOGGING"
debugMalloc="-UDEBUG_MALLOC"
selfManagedMemory="-UNO_SELF_MANAGING"
qmmes="-UQMMMES"
qmmis="-UQMMIS"
qmmae="-DQMMAE=True"
qmmib="-UQMMIB"
checkForLeaks="-UCHECK_LEAKING"
runCtest=false
enableMonitoring="-UENABLE_MONITORING"
tvlpath="-UTVL_PATH"
testAll="-DCTEST_ALL=False"
testMemory="-DCTEST_MEMORY=False"
testMorph="-DCTEST_MORPHING=False"
testOps="-DCTEST_OPERATORS=False"
testPers="-DCTEST_PERSISTENCE=False"
testStorage="-DCTEST_STORAGE=False"
testUtils="-DCTEST_UTILS=False"
testVectors="-DCTEST_VECTOR=False"
buildAll="-DBUILD_ALL=False"
buildCalibration="-DBUILD_CALIB=False"
buildExamples="-DBUILD_EXAMPLES=False"
buildMicroBms="-DBUILD_MICROBMS=False"
buildSSB="-DBUILD_SSB=False"
avx512="-DCAVX512=False"
avxtwo="-DCAVXTWO=False"
sse4="-DCSSE=False"
rapl="-DCRAPL=False"
odroid="-DCODROID=False"
neon="-DCNEON=False"
sve="-DCSVE=False"
target="all"
vbpLimitRoutinesForSSBSF1="-UVBP_LIMIT_ROUTINES_FOR_SSB_SF1"
vbpLimitRoutinesForSSBSF100="-UVBP_LIMIT_ROUTINES_FOR_SSB_SF100"
projectAssumePrepared="-UPROJECT_ASSUME_PREPARED"

numCores=`nproc`
if [ $numCores != 1 ]
then
    makeParallel="-j$((numCores - 1))"

fi


while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -h|--help)
    printHelp
    exit 0
    ;;
    -nl|--noLog)
    logging="-DNOLOGGING=True"
    shift # past argument
    ;;
    -debM|--debug-Malloc)
    debugMalloc="-DDEBUG_MALLOC=True"
    shift # past argument
    ;;
    -noSelfManaging)
    selfManagedMemory="-DNO_SELF_MANAGING=True"
    shift # past argument
    ;;
    -lc|--leakCheck)
    checkForLeaks="-DCHECK_LEAKING=True"
    shift # past argument
    ;;
    -queryMinMemEx)
    if ! is_power_of_two $2; then
        echo "queryMinMemEx is not a power of 2, exiting."
        exit -1
    fi
    qmmes="-DQMMMES=$2"
    shift
    shift
    ;;
    -queryInitSize)
    qmmis="-DQMMIS=$2"
    shift
    shift
    ;;
    -queryDisallowExpand)
    qmmae="-UQMMAE"
    shift # past argument
    ;;
    -queryInitBuffers)
    qmmib="-DQMMIB=True"
    shift # past argument
    ;;
    --alignment)
    if ! is_power_of_two $2; then
        echo "Memory Manager alignment is not a power of 2, exiting"
        exit -1
    fi
    setMemoryAlignment="-DMMGR_ALIGN=$2"
    shift
    shift
    ;;
    -deb|--debug)
    buildModeSet=$((buildModeSet + 1))
    buildMode="-DCMAKE_BUILD_TYPE=Debug"
    shift # past argument
    ;;
    -rel|--release)
    buildModeSet=$((buildModeSet + 1))
    buildMode="-DCMAKE_BUILD_TYPE=Release"
    shift # past argument
    ;;
    -hi|--HighPerf)
    buildModeSet=$((buildModeSet + 1))
    buildMode="-DCMAKE_BUILD_TYPE=HighPerf"
    shift # past argument
    ;;
    -mon|--enable-monitoring)
    enableMonitoring="-DENABLE_MONITORING=True"
    shift # past argument
    ;;
    -tA|--testAll)
    runCtest=true
    testAll="-DCTEST_ALL=True"
    shift # past argument
    ;;
    -tMm|--testMem)
    runCtest=true
    testMemory="-DCTEST_MEMORY=True"
    shift # past argument
    ;;
    -tMo|--testMorph)
    runCtest=true
    testMorph="-DCTEST_MORPHING=True"
    shift # past argument
    ;;
    -tOp|--testOps)
    runCtest=true
    testOps="-DCTEST_OPERATORS=True"
    shift # past argument
    ;;
    -tPe|--testPers)
    runCtest=true
    testPers="-DCTEST_PERSISTENCE=True"
    shift # past argument
    ;;
    -tSt|--testStorage)
    runCtest=true
    testStorage="-DCTEST_STORAGE=True"
    shift # past argument
    ;;
    -tUt|--testUtils)
    runCtest=true
    testUtils="-DCTEST_UTILS=True"
    shift # past argument
    ;;
        -bA|--buildAll)
    buildAll="-DBUILD_ALL=True"
    shift # past argument
    ;;
        -bCa|--buildCalibration)
    buildCalibration="-DBUILD_CALIB=True"
    shift # past argument
    ;;
        -bEx|--buildExamples)
    buildExamples="-DBUILD_EXAMPLES=True"
    shift # past argument
    ;;
        -bMbm|--buildMicroBms)
    buildMicroBms="-DBUILD_MICROBMS=True"
    shift # past argument
    ;;
    -bSSB|--buildSSB)
    if ! is_int $2; then
        echo "-bSSB or --buildSSB must be followed by the scale factor as an integer"
        exit -1
    fi
    buildSSB="-DBUILD_SSB=$2"
    shift
    shift # past argument
    ;;
    -avx512)
    avx512="-DCAVX512=True"
        avxtwo="-DCAVXTWO=True"
    shift # past argument
    ;;
    -avxtwo)
        avxtwo="-DCAVXTWO=True"
    shift # past argument
    ;;
        -sse4)
        sse4="-DCSSE=True"
    shift # past argument
    ;;
        -armneon)
        neon="-DCNEON=True"
    shift # past argument
    ;;
        -armsve)
        sve="-DCSVE=True"
    shift # past argument
    ;;
    -tVt|--testVectoring)
    runCtest=true
    testVectors="-DCTEST_VECTOR=True"
    shift # past argument
    ;;
        -rapl)
        rapl="-DCRAPL=True"
    shift # past argument
    ;;
        -odroid)
        odroid="-DCODROID=True"
    shift # past argument
  ;;
    --tvl)
    tvlpath="-DTVL_PATH=$2"
    shift
  shift
    ;;
    --target)
    target=$2
    shift
    shift
        ;;
        --vbpLimitRoutinesForSSBSF1)
        vbpLimitRoutinesForSSBSF1="-DVBP_LIMIT_ROUTINES_FOR_SSB_SF1=True"
        shift # past argument
        ;;
        --vbpLimitRoutinesForSSBSF100)
        vbpLimitRoutinesForSSBSF100="-DVBP_LIMIT_ROUTINES_FOR_SSB_SF100=True"
        shift # past argument
        ;;
        --projectAssumePrepared)
        projectAssumePrepared="-DPROJECT_ASSUME_PREPARED=True"
        shift # past argument
        ;;
    *)
    optCatch='^-j'
    if ! [[ $1 =~ $optCatch ]]
    then
        printf "%s: Unknown option\n" $1
        if [[ $1 == *"="* ]]; then
          echo "Value assignment is done without the equality sign, use a space! E.g. --alignment 256"
        fi
        exit -1
    else
        re='^-j[1-9][0-9]*$'
        if ! [[ $1 =~ $re ]];
        then
           printf "j: Invalid Syntax. Use -j[1-9][0-9]+\n" >&2; exit 1
        else
            makeParallel=$1
        fi
    fi
    shift # past value
    ;;
esac
done;

if [ $buildModeSet -gt 1 ] || [ $buildModeSet -eq 0 ]
then
    printf "BuildMode not set correctly.\n"
    printHelp
    exit 1
fi

printf "Using buildMode: $buildMode and make with: $makeParallel $target\n"

if [ "$runCtest" = true ] ; then
    addTests="-DRUN_CTESTS=True $testAll $testMemory $testMorph $testOps $testPers $testStorage $testUtils $testVectors $avx512 $avxtwo $odroid $rapl $neon $sve $tvlpath"
    echo "AddTest String: $addTests"
else
    addTests="-DRUN_CTESTS=False"
fi
addBuilds="$buildAll $buildCalibration $buildExamples $buildMicroBms $buildSSB"

set -e # Abort the build if any of the following commands fails.
mkdir -p build
cmake -E chdir build/ cmake $buildMode $logging $selfManagedMemory $qmmes $qmmis $qmmae $qmmib $debugMalloc $checkForLeaks $setMemoryAlignment $enableMonitoring $addTests $addBuilds $avx512 $avxtwo $sse4 $odroid $rapl $neon $sve $vbpLimitRoutinesForSSBSF1 $vbpLimitRoutinesForSSBSF100 $tvlpath $projectAssumePrepared -G 'Unix Makefiles' ../
make -C build/ VERBOSE=1 $makeParallel $target
set +e

if [ "$runCtest" = true ] ; then
    cd build && ctest --output-on-failure #--extra-verbose
else
    echo "No tests to be run"
fi
