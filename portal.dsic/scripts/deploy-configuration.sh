#!/bin/bash

workers=$(cat ${SPARK_HOME}/conf/workers | grep -v "^#" | grep -v "^$" | awk '{ print $1 }')

for wn in ${workers}
do
	echo "${wn}"

	case ${wn} in
		localhost|teaa-master-ubuntu22.dsicv.upv.es)
			echo "      Nothing to do!"
			;;
		*)
			for dir in bin cluster scripts
			do
				echo rsync -avHC --delete ${HOME}/teaa/${dir}     ${wn}:teaa/
				rsync -avHC --delete ${HOME}/teaa/${dir}     ${wn}:teaa/  #&
			done
			;;
	esac
done
