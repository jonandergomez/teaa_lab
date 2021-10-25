#!/bin/bash

workers=$(cat etc_hosts.cluster | grep worker | grep -v "^#" | grep -v "^$" | awk '{ print $NF }')
cluster_base_dir="${HOME}/teaa/disk"

for wn in ${workers} 
do
	echo "${wn}"

	ssh -oStrictHostKeyChecking=no root@${wn} apt install -y libkryo-java libkryo-serializers-java 
	ssh -oStrictHostKeyChecking=no root@${wn} apt install -y python3  python3-numpy
	ssh -oStrictHostKeyChecking=no root@${wn} apt install -y python3-sklearn python3-sklearn-lib python3-sklearn-pandas
    #scp etc_hosts.cluster root@${wn}:/etc/hosts

	#ssh ubuntu@${wn} echo $PATH # sanity check
	#ssh ubuntu@${wn} which python3 # sanity check
	#ssh ubuntu@${wn} which pip3 # sanity check
	#ssh ubuntu@${wn} ls -l /home/ubuntu/anaconda3/bin/pip3 # sanity check

	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} pip3 install numpy
	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} conda update --all 

    #scp ~/.profile ubuntu@${wn}:.profile
    #scp ~/.bashrc ubuntu@${wn}:.bashrc
done
