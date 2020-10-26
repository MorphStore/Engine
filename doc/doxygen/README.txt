###################################################################################################
#   This ReadMe is about building the doxygen documentation of the ERIS data management system    #
###################################################################################################

#==============================#
# Requirements                 #
#==============================#
1. Doxygen
   - Download and install doxygen
     http://www.stack.nl/~dimitri/doxygen/
     http://www.stack.nl/~dimitri/doxygen/download.html
2. Graphviz
   - Download and install Graphviz
     http://www.graphviz.org/
	 http://www.graphviz.org/Download.php
3. Make sure the binaries of both are available for command line (Windows: bin folder in $PATH variable)

#==============================#
# Structure                    #
#==============================#
1. All doxygen related files are located in doxygen-subfolder (except start scripts)
2. doxygen/doxygen.eris.config           : specific config for doxygen
3. doxygen/pages/                        : manual created pages
4. doxygen/style/style.css               : additional CSS
5. doxygen/documentation/                : all generated files will be written in this folder (by default)
                                         : WARNING: Do not commit this folder into eris-git!
6. doxygen/documentation/html/index.html : start page of the generated documentation

#==============================#
# Build documentation          #
#==============================#
1. Move to eris root directory
2. Run doxygen by one of these options
   1. Startscripts
      - $ ./doxy.bat (Windows)
	  - $ ./doxy.sh  (everything else)
   2. Manual
      - $ doxygen doxygen/doxygen.eris.config
	  
#==============================#
# Last step                    #
#==============================#
Open the indexfile with a browser of your choice ;)
doxygen/documentation/html/index.html