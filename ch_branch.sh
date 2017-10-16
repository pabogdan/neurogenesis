#!/bin/bash
echo "Script to change branches in the following repositories:" 
echo $@

branch=$1
shift
for arg in $@
do
if [[ -d "$arg" ]]
	then
	echo "Changing branch in $arg"
	cd $arg && git checkout $branch && echo "--Checkout complete--"
	cd ..
fi
done