#!/bin/bash
###==================================================================================================================###
### Parse runtime parameters                                                                                         ###
###==================================================================================================================###
function parse_arguments() {
  for param in "$@"; do
    case $param in
      -h|--help)
        print_help_message
        exit;;
      -a|--alias)
        cat doc/doxygen/doxygen.alias.conf | grep "^ALIAS"
        exit 0 ;;
      --lite|-l)
        LITE=1
        OUTPUT_PATH_EXT=lite ;;
      --extremelite|-el)
        EXTREME_LITE=1
        OUTPUT_PATH_EXT=extreme_lite ;;
      --opendoc|-o)
        OPEN_DOC=1 ;;
      --opendocl|-ol)
        OPEN_DOC=1
        OUTPUT_PATH_EXT=_lite ;;
      --opendocel|-oel)
        OPEN_DOC=1
        OUTPUT_PATH_EXT=extreme_lite ;;
      --output*)
        OUTPUT_PATH=${param#--output=} ;;
      --config*)
        CONFIG_FILE=${param#--config=} ;;
      --copy*)
        COPY=${param#--copy=} ;;
      --no-color|-nc)
  #     NO_COLOR=1 ;;
        reset_color_codes ;;
      --install)
        install_dependencies
        exit;;
      --uninstall)
        uninstall_dependencies
        exit;;
      *)
        DXY_PARAM["$DXY_PARAM_CNT"]="$param"
        ((DXY_PARAM_CNT++)) ;;
    esac
  done
}
#if [[ $NO_COLOR -eq 0 ]]; then
#  set_color_codes
#fi
