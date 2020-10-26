#!/bin/bash

# Script to generate MorphStore documentation
# run in MorphStore/Engine directory only

DOXY_ROOT="$(dirname $0)/doc/doxygen"


### Paths ###
# Path to doxygen config file
CONFIG_FILE=$DOXY_ROOT/doxygen.morphstore.conf
ALIAS_FILE=$DOXY_ROOT/doxygen.alias.conf
# Path to the generated output
OUTPUT_PATH="../Documentation"
OUTPUT_PATH_EXT="full"
# Path to your default browser executable
# alternatively set an environment variable in your .bashrc like this: export DEFAULT_BROWSER="{path to browser}"
#DEFAULT_BROWSER=

SHELL_SCRIPTS="$DOXY_ROOT/shell"


###===== parameters =================================================================================================###
HELP=0
LITE=0
EXTREME_LITE=0
OPEN_DOC=0
NO_COLOR=0
COPY=""
INSTALL=0
UNINSTALL=0

DXY_PARAM_CNT=0
DXY_PARAM[0]=""


###===== import color codes =========================================================================================###
source $SHELL_SCRIPTS/ColorCodes.sh
###===== import print functions =====================================================================================###
source $SHELL_SCRIPTS/PrintFunctions.sh
###===== import un-/install functions ===============================================================================###
source $SHELL_SCRIPTS/InstallDependencies.sh
###===== import help message ========================================================================================###
source $SHELL_SCRIPTS/HelpMessage.sh
###===== parse script arguments =====================================================================================###
source $SHELL_SCRIPTS/ParseArguments.sh

parse_arguments $@
print_first_use_message

echo "Output path: ${OUTPUT_PATH}"

### Add extension to output path
OUTPUT_PATH="${OUTPUT_PATH}/${OUTPUT_PATH_EXT}"

# Windows path fix
if [ ! -z $WINDOWS ] && [[ $OUTPUT_PATH == "/"* ]]; then
  disk=${OUTPUT_PATH:0:2}
  case $disk in
    /c)
      OUT="C:"${OUTPUT_PATH:2};;
    /d)
      OUT="D:"${OUTPUT_PATH:2};;
  esac
else
  OUT="$OUTPUT_PATH"
fi



### Open documentation ### =============================================================================================
if [ "$OPEN_DOC" == "1" ]; then
    if [ "$DEFAULT_BROWSER" == "" ]; then
        error "No default browser defined. Add the path to this script or use an environment variable called DEFAULT_BROWSER."
    else
        if [ ! -e "$DEFAULT_BROWSER" ]; then
            error "Browser binary ${YELLOW}${DEFAULT_BROWSER}${RED} not found."
            exit 0
        fi
        if [ ! -x "$DEFAULT_BROWSER" ]; then
            error "Browser binary ${YELLOW}${DEFAULT_BROWSER}${RED} not executable."
            exit 0
        fi

        info "Opening documentation at ${YELLOW}${OUTPUT_PATH}${NOC}"
        "$DEFAULT_BROWSER" "file://${OUT}/html/index.html"
    fi
    exit 0
fi


###==================================================================================================================###
### Build documentation (default)                                                                                    ###
###==================================================================================================================###
info "Use ${CYAN}doxy.sh -h$NOC for more options.\n"

### ===== Check generation tools
if [ -z "$(command -v doxygen)" ]; then
    CMD_ERROR=1
    error "Package doxygen not found."
fi

if [ -z "$(command -v dot)" ]; then
    CMD_ERROR=1
    error "Package dot not found."
fi

if [ -z "$(command -v sass)" ]; then
    CMD_ERROR=1
    error "Package sass not found."
fi

if [ -n "$CMD_ERROR" ]; then
    error "Please install missing dependencies with $ sudo ./doxy.sh --install"
    exit 1
fi



### Check for config file ### ==========================================================================================
if [ ! -f $CONFIG_FILE ]; then
    warn "Config file $CONFIG_FILE not found. Usinge default config file."
    CONFIG_FILE=$DOXY_ROOT/doxygen.morphstore.conf.default
fi

if [ ! -f $CONFIG_FILE ]; then
    error "Config file ${YELLOW}${CONFIG_FILE}${NOC} not found."
    exit 1
fi




### Call setup script to ensure dependencies ### =======================================================================
#info "${RED}#=== ${GREEN}Call setup.sh${RED} =======================================================================================#${NOC}"
#if [ "$NO_COLOR" == "1" ]; then
#    sh setup.sh -nc
#else
#    sh setup.sh
#fi
#info "${RED}#=== ${GREEN}End setup.sh${RED} ========================================================================================#${NOC}\n"

### Generation ### =====================================================================================================
info "Starting doxygen using $YELLOW$CONFIG_FILE$NOC as config file.\n"
info "Output will be written to ${YELLOW}${OUTPUT_PATH}${NOC}."

# Create error output file
touch $DOXY_ROOT/errors.txt

# create output path
mkdir -p "${OUTPUT_PATH}/html"

# read config file
CONFIG=$(<$CONFIG_FILE)$(<$ALIAS_FILE)


#for line in "$CONFIG"; do
#  echo "$line"
#done


#CONFIG="${CONFIG}"$'\n'"OUTPUT_DIRECTORY = ${OUT}"
#echo -E "$CONFIG"
#exit

## == lite mode == ##	
if [ "$LITE" == "1" ]; then
  info "Generation is running in ${CYAN}~~ lite mode ~~${NOC}"

  (
    echo "$CONFIG"
    echo ""
#        echo "SOURCE_BROWSER         = NO"
    echo "INLINE_SOURCES         = NO"
    echo "CALL_GRAPH             = NO"
    echo "INLINE_INHERITED_MEMB  = NO"
    echo "CLASS_DIAGRAMS         = NO"
    echo "HAVE_DOT               = NO"
    echo "REFERENCED_BY_RELATION = NO"
    echo "REFERENCES_RELATION    = NO"
#        echo "SEPARATE_MEMBER_PAGES  = NO"
    echo "VERBATIM_HEADERS       = NO"
    echo "OUTPUT_DIRECTORY       = ${OUTPUT_PATH}"
    for param in ${DXY_PARAM[*]}; do
        echo "$param"
    done
  ) | doxygen -
fi


## == extreme lite mode == ##	
if [ "$EXTREME_LITE" == "1" ]; then
  info "Generation is running in $RED~~~ extreme lite mode ~~~$NOC"
  (
    echo "$CONFIG"
    echo "INPUT = doc/doxygen/pages/"
    echo "OUTPUT_DIRECTORY = ${OUTPUT_PATH}"
    for param in ${DXY_PARAM[*]}; do
        echo "$param"
    done
  ) | doxygen -
fi


## == normal mode == ##	
if [ "$LITE" == "0" ] && [ "$EXTREME_LITE" == "0" ]; then
    info "Generation is running in $GREEN~ normal mode ~$NOC (this could take several minutes)"

#    (
##        cat $CONFIG_FILE
#        for line in $CONFIG; do
#            echo "$line"
#        done
#        echo "OUTPUT_DIRECTORY = ${OUT}"
#        for param in ${DXY_PARAM[*]}; do
#            echo "$param"
#        done
#    ) | doxygen -


  CONFIG="${CONFIG}"$'\n'"OUTPUT_DIRECTORY = ${OUTPUT_PATH}"
  echo "$CONFIG" | doxygen -
fi

### Compile stylsheets### ==============================================================================================
#OLD_PWD="$PWD"
#
#cd doxygen/style
#files=("scss/*.scss")
#
#for file in $files; do
#    fileout=${file##*/}
#    fileout=${fileout%.scss}
#    sass "$file" "${fileout}.css"
#done
#
#cd "$OLD_PWD"

sass $DOXY_ROOT/style/scss/main.scss $DOXY_ROOT/style/main.css

### Copy stylsheets & scripts ### ======================================================================================
mkdir ${OUTPUT_PATH}/html/images/
mkdir ${OUTPUT_PATH}/html/style/
mkdir ${OUTPUT_PATH}/html/style/search/
cp $DOXY_ROOT/images/* ${OUTPUT_PATH}/html/images/
cp $DOXY_ROOT/style/*.css ${OUTPUT_PATH}/html/style/ > /dev/null
cp $DOXY_ROOT/style/search/* ${OUTPUT_PATH}/html/style/search/
cp -r $DOXY_ROOT/script/* ${OUTPUT_PATH}/html/script/


### Print results ### ==================================================================================================
info "Generation complete.\n"

echo "${RED}#=== ${GREEN}Warnings and Errors${RED} =================================================================================#${NOC}";
if (! (cat $DOXY_ROOT/errors.txt | grep -v "resolve reference\|multiple use of section label")); then
  echo "[SUCCESS] No warnings or errors"
fi

echo ""
info "${RED}#=== ${GREEN}Duplicate Labels${RED} ====================================================================================#${NOC}";
if (! (cat $DOXY_ROOT/errors.txt | grep "multiple use of section label")); then
  info "${GREEN}SUCCESS. No duplicates"
fi

echo ""
info "${RED}#=== ${GREEN}Unresolved References${RED} ===============================================================================#${NOC}";
if (! (cat $DOXY_ROOT/errors.txt | grep "resolve reference")); then
  info "${GREEN}SUCCESS. No unresolved references"
fi


### Create copy of the result ### ======================================================================================
if [ ! "$COPY" == "" ]; then
  echo ""
  info "Copying files to ${YELLOW}${COPY}${NOC}"
  cp -r "$OUTPUT_PATH/html/" "$COPY"
fi
