#!/bin/bash

workers=$(cat ${SPARK_HOME}/conf/workers | grep -v "^#" | grep -v "^$" | awk '{ print $1 }')
cluster_base_dir="${HOME}/teaa/disk"

for wn in ${workers} teaa-master-ubuntu22.dsicv.upv.es
do
	echo "${wn}"

	case ${wn} in
		localhost|teaa-master-ubuntu22*)
			#echo "Nothing to do!"
			;;
		*)
			#ssh ${wn} mkdir -p .ssh
			#scp ${HOME}/.ssh/authorized_keys ${wn}:.ssh/authorized_keys
			#ssh -oStrictHostKeyChecking=no ${wn} chsh -s /bin/bash
			;;
	esac


	#scp ${HOME}/.profile ${wn}:.profile # &
	#scp ${HOME}/.bashrc  ${wn}:.bashrc  # &

	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/logs"
#	ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/hdfs/namenode"
#	ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/hdfs/datanode"
	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/dirs/local" 
	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/dirs/logs"
	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/dirs/mr-history/tmp"
	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/dirs/mr-history/done"
	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/dirs/spark/local"
	#ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/dirs/spark/work"
#
	#ssh -oStrictHostKeyChecking=no ${wn} ls -la ${cluster_base_dir}/
	#ssh -oStrictHostKeyChecking=no ${wn} ls -la ${cluster_base_dir}/dirs/
	#ssh -oStrictHostKeyChecking=no ${wn} ls -la ${cluster_base_dir}/hdfs/

done
