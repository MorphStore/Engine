\page quickStart Download and compile
   
Download
========

The source and build-script of MorphStore can be found in the github repository https://github.com/MorphStore/Engine.git.
To get a local copy, open a terminal (linux) or a git bash (depends on the git client you installed in windows), 
change to the directory, where you want your copy to be, and execute the following command:

~~~{.sh}
git clone https://github.com/MorphStore/Engine.git
~~~

 
Compile
=======

Since all our test systems run a derivate of Linux, we only provide building scripts for linux. However, in Windows 10, 
a linux subsystem can be installed, which we have also used successfully to compile and run MorphStore.

Ensure that you have the following tools installed before trying to build:
- gcc >= version 8.2
- g++ >= version 8.2
- cmake >= version 3.10

Older versions may not build all test cases. Note that C++14 is necessary.


To facilitate building and testing MorphStore, there is a script <i>build.sh</i> in the root folder.
A complete list with all available build options can be shown by running

~~~{.sh}
./build.sh -h
~~~

For starters, the following command can be used for an initial test of the setup:

~~~{.sh}
./build.sh -deb -tQ
~~~

This builds some example queries in debug mode and runs them. The source code of these queries can be found in the folder src/examples.
They are runnig in scalar mode. Thus, every system providing C++14 support should be able to build and run them regardless of any (not) 
available vector extensions.

Test Vector Extensions
======================

To enable the support of vector extensions, build.sh provides several flags:

- sse4
- avxtwo
- avx512

For some architectures, gcc enbales sse or even avx by default. But for others, the available extensions have to be passed explicitly. 
This cared for by the build script if one of the above mentioned flags is set. Every flag passes the according extension and all older 
extensions, e.g. -avxtwo passes the flags for axv2 and sse4 support.

Using these flags, vectorized operators and queries can be build. Some operators, which are called with sse and avx2 can be found in 
the folder test/vector/operators. They can be built and run with

~~~{.sh}
./build.sh -deb -tVt -avxtwo
~~~  

Again, this builds in debug mode. For release mode (-O2) use -rel, and for high performance mode (-O3 and link time optimization) use -hi.

next: \ref testCases 