#!/bin/bash
function printHelp {
	echo "build.sh -buildMode [-loggerControl] [-memory] [-jN]"
	echo "buildMode:"
	echo "	-rel|--release"
	echo "	     Release mode"
	echo "	-deb|--debug"
	echo "	     Debug mode"
	echo ""
	echo "loggerControl:"
	echo "	-nl|--noLog"
	echo "	     Completely removes all calls to the logger."
	echo ""
	echo "memory:"
	echo "	-debM"
	echo "	     Prints calls to the Memory manager on the debug channel."
	echo "	-noSelfManaging"
	echo "	     Deactivates the memory hooks and enables standard C++ memory management."
	echo "	-queryMinMemEx 2^x"
	echo "	     Sets the minimum chunk reallocation size in bytes for the Query Memory Manager. The value should be a power of two."
	echo "	-lc|--leakCheck"
	echo "	     Makes the MemoryManager more aware of possible memory leaks."
	echo "	--alignment 2^x"
	echo "	     Sets the byte alignment for the MemoryManager"
	echo ""
	echo "	-jN:"
	echo "	     N > 0 sets the number of parallel make jobs"
}

buildType=""
makeParallel="1"

buildModeSet=0

logging="-UNOLOGGING"
debugMalloc="-UDEBUG_MALLOC"
selfManagedMemory="-UNO_SELF_MANAGING"
qmmes="-UQMMMES"
checkForLeaks="-UCHECK_LEAKING"

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
	qmmes="-DQMMMES=$2"
	shift
	shift
	;;
	--alignment)
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
	*)
	optCatch='^-j'
	if ! [[ $1 =~ $optCatch ]]
	then
		printf "%s: Unknown option\n" $1
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

printf "Using buildMode: $buildMode and make with: $makeParallel\n"

mkdir -p build
cmake -E chdir build/ cmake $buildMode $logging $selfManagedMemory $qmmes $debugMalloc $checkForLeaks $setMemoryAlignment -G "Unix Makefiles" ../
make -C build/ VERBOSE=1 $makeParallel
