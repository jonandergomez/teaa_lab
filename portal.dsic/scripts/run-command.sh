#!/bin/bash

workers=$(cat etc_hosts.cluster | grep worker | grep -v "^#" | grep -v "^$" | awk '{ print $NF }')
cluster_base_dir="${HOME}/teaa/disk"

for wn in ${workers} 
do
	echo "${wn}"

	ssh -oStrictHostKeyChecking=no root@${wn} apt install -y libkryo-java libkryo-serializers-java 
done
