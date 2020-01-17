# Welcome to MorphStore!

## Quick Start
Get the source with *git clone https://github.com/MorphStore/Engine.git*

Make sure to have __gcc/g++ >= version 8__ and __cmake >= version 3.10__ installed.

Build a simple test query by *./build.sh –rel –bEx --target select_sum_query* from the root folder. This example uses no vectorization or compression, and runs without the Template Vector Library. It can be used to test the basic functionality of MorphStore, e.g. memory management, and the general workflow.
The binary can be found in build/src/examples/.

Call *./build.sh -h* for more options.

~~See if it works by calling ./build.sh -deb -tQ from the root folder.~~
-> **Returning users** We changed the build script a few times in the last quarter of 2019, so this old call won't work anymore after checking out the current version.


## Documentation
Build and open the Documentation:

- Install doxygen
- "cd doc/doxygen"
- "doxygen morphStory"
- open doc/doxygen/html/index.html

### Compression
Dear Sigmod'19 visitors, the morphing operators and compression formats can be found in include/core/morphing. The code contains many useful comments.

### Template Vector Library (TVL)
Dear Cidr'20 visitors, the TVL is currently an integrated part of morphstore. We are working on a stand-alone version. Until then, you can find the TVL in include/vector. 
You are also welcome to check out the branch TVLtesting, which contains more (but probably unstable) functionality.
