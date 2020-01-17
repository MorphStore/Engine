# Welcome to MorphStore!

## Quick Start
Get the source with *git clone https://github.com/MorphStore/Engine.git*

Make sure to have __gcc/g++ >= version 8__ and __cmake >= version 3.10__ installed.

See if it works by calling *./build.sh -deb -tQ* from the root folder.

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
