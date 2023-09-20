#!/bin/bash

workers=$(cat etc_hosts.cluster | grep worker | grep -v "^#" | grep -v "^$" | awk '{ print $NF }')
cluster_base_dir="${HOME}/teaa/disk"

for wn in ${workers} 
do
	echo "${wn}"

	#ssh -oStrictHostKeyChecking=no root@${wn} apt install -y libkryo-java libkryo-serializers-java 
	#ssh -oStrictHostKeyChecking=no root@${wn} apt install -y python3  python3-numpy
	#ssh -oStrictHostKeyChecking=no root@${wn} apt install -y python3-sklearn python3-sklearn-lib python3-sklearn-pandas
    	#scp etc_hosts.cluster root@${wn}:/etc/hosts

	#rsync -avn --delete /opt/anaconda3/ ubuntu@${wn}:/opt/anaconda3/

	# sanity checks
	ssh ubuntu@${wn} 'echo $PATH'
	ssh ubuntu@${wn} 'which python3'
	ssh ubuntu@${wn} 'which pip3'
	ssh ubuntu@${wn} 'ls -l /opt/anaconda3/bin/pip3'

	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} pip3 install numpy
	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} conda update --all 

	#scp ~/.profile ubuntu@${wn}:.profile
	#scp ~/.bashrc ubuntu@${wn}:.bashrc
done
