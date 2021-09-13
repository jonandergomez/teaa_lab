#!/bin/bash

workers=$(cat etc_hosts.cluster | grep worker | grep -v "^#" | grep -v "^$" | awk '{ print $NF }')
cluster_base_dir="${HOME}/teaa/disk"

for wn in ${workers} 
do
	echo "${wn}"

	#ssh -oStrictHostKeyChecking=no root@${wn} apt install -y libkryo-java libkryo-serializers-java 
	#ssh -oStrictHostKeyChecking=no root@${wn} apt install -y python3  python3-numpy
	ssh -oStrictHostKeyChecking=no root@${wn} apt install -y python3-sklearn python3-sklearn-lib python3-sklearn-pandas
	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} pip3 install numpy
    #scp ~/.profile ubuntu@${wn}:.profile
	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} conda update --all 
done
