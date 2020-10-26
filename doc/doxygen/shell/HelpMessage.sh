#!/bin/bash
###==================================================================================================================###
### Welcome Message                                                                                                  ###
###==================================================================================================================###
function print_first_use_message() {
  if [ ! -f "$DOXY_ROOT/tmp/firstrun" ]; then
    mkdir -p "$DOXY_ROOT/tmp/"
    touch $DOXY_ROOT/tmp/firstrun
    info "Oh hi. Nice to see a new face around :)"
    info "This seems to be your first run, so I will show you some helpful information:"
    info "Call \"doxy.sh -h\" for help"
    info "Call this script without parameters to trigger the default doxygen generation."
    info "Call \"doxy.sh --output=<path>\" to change the target output location"
    info "Set the path to your prefered browser executable in the variable DEFAULT_BROWSER. Then you can call \"doxy.sh -od\" to quick open the generated documentation."
  fi
}
###==================================================================================================================###
### Help page                                                                                                        ###
###==================================================================================================================###
function print_help_message() {
    info "Usage: doxy [OPTION | DOXYGEN_PARAMETER]..."
    info "Generates the documentation of ERIS using doxygen"
    info ""
    info "Without any options, default generation is triggered"
    info ""
    info "Options:"
    info "  -h,         --help              Display this help page"
    info "              --output=<path>     Changes the output directory"
    info "              --config=<file>     Changes the doxygen config file"
    info "              --copy=<path>       After generation the contents of <ouput_path>/html/ are copied to this path"
    info "  -nc,        --no-color          Turn off colorization"
    info ""
    info "Generation modes:"
    info "  -l,         --lite              Reduces cpu and io intensive operations for faster custom page generation"
    info "  -el,        --extremelite       Only generate custom pages (links to classes/members will not work)"
    info " Speed: normal (several minutes), lite (some seconds), extreme lite (instant)"
    info ""
    info ""
    info "Quick open documentation:"
    info "To use these DEFAULT_BROWSER has to be set"
    info "  -od,        --opendoc           Open documentation generated in normal mode"
    info "  -odl,       --opendocl          Open documentation generated in lite mode"
    info "  -odel,      --opendocel         Open documentation generated in extreme lite mode"
    info ""
    info ""
    info "Dependencies"
    info "              --install           Install dependend packages to generate this documentation (sudo required)"
    info "              --uninstall         Uninstall packages (sudo required) packages are uncritical"
    info ""
    info ""
    info "To override a doxygen parameter, just pass it as parameter (no spaces), e.g.:"
    info "  doxy.sh QUIET=NO    # turn on doxygen generation output"
}
