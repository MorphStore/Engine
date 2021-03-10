#!/bin/bash
# if [ "$EUID" -ne 0 ]
  # then echo "Please run as root"
  # exit
# fi

which cat
if [[ $? == 1 ]]; then echo "error"; fi

function info() {
  echo $@
}



exit

while(true); do
	echo -n "This will remove following packages: doxygen graphviz sass npm. Are you sure? [y/n]: "
	read accept
	if [ "$accept" = "y" ]; then break; fi
	if [ "$accept" = "n" ]; then echo "Exit."; exit; fi
done

echo "Accepted"

