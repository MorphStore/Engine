#!/bin/bash
###==================================================================================================================###
### Install dependencies                                                                                             ###
###==================================================================================================================###
function install_dependencies() {
	if [ "$EUID" -ne 0 ]; then
		echo "Please run with root privileges to install dependencies."
		exit
	fi
	#// TODO: support different package manager

	while(true); do
		echo -n "This will install following packages: doxygen graphviz sass npm. Are you sure? [y/n]: "
		read accept
		if [ "$accept" = "y" ]; then break; fi
		if [ "$accept" = "n" ]; then echo "Exit."; exit; fi
	done
    # try to install dependencies
  if [ -z "$(command -v apt)" ]; then
    warn "Command apt not found. Skip installing dependencies."
		exit
	fi

	if [ -z "$(command -v npm)" ]; then
		info "Installing npm."
    apt install -y npm
    if [[ $? == 1 ]]; then error "Could not install npm."; exit; fi
  else
    info "npm already installed."
	fi

	if [ -z "$(command -v doxygen)" ]; then
    info "Installing doxygen."
    apt install -y doxygen
    if [[ $? == 1 ]]; then error "Could not install doxygen."; exit; fi
  else
    info "doxygen already installed."
  fi

	if [ -z "$(command -v dot)" ]; then
    info "Installing graphviz."
    apt install -y graphviz
    if [[ $? == 1 ]]; then error "Could not install graphviz."; exit; fi
	else
	  info "graphviz already installed."
  fi

	if [ -z "$(command -v sass)" ]; then
    info "Installing sass."
    npm install -g sass
    if [[ $? == 1 ]]; then error "Could not install sass."; exit; fi
	else
	  info "sass already installed."
	fi

	info "All dependencies installed successfully."
}

###==================================================================================================================###
### Uninstall dependencies                                                                                           ###
###==================================================================================================================###
function unsinstall_dependencies() {
	if [ "$EUID" -ne 0 ]; then
		echo "Please run with root privileges to remove dependencies."
		exit
	fi
    APT_CHECK=$(command -v apt)
    NPM_CHECK=$(command -v npm)

	while(true); do
		echo -n "This will remove following packages: doxygen graphviz sass npm. Are you sure? [y/n]: "
		read accept
		if [ "$accept" = "y" ]; then break; fi
		if [ "$accept" = "n" ]; then echo "Exit."; exit; fi
	done

  # try to install dependencies
  if [ -z "$(command -v apt)" ]; then
    echo "[${LYELLOW}WARNING${NOC}] Command apt not found. Skip uninstalling dependencies."
		exit
	fi

	if [ -z "$(command -v npm)" ]; then
        echo "[${LYELLOW}WARNING${NOC}] Command npm not found. Skip uninstalling dependencies."
		exit
	fi

	echo "[${LGREEN}   INFO${NOC}] Uninstalling doxygen."
	apt remove -y doxygen
	echo "[${LGREEN}   INFO${NOC}] Uninstalling graphviz."
	apt remove -y graphviz
	echo "[${LGREEN}   INFO${NOC}] Uninstalling sass."
	npm uninstall -g sass
	echo "[${LGREEN}   INFO${NOC}] Uninstalling npm."
	apt remove -y npm
	echo "[${LGREEN}   INFO${NOC}] Done."
}
