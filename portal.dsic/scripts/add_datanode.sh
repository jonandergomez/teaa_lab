#!/bin/bash


cluster_base_dir="teaa/disk"

for n in $*
do
	wn="teaa-worker${n}-ubuntu22.dsicv.upv.es"
	echo "${wn}"

	ssh -oStrictHostKeyChecking=no ubuntu@${wn} teaa/cluster/hadoop/bin/hdfs --daemon stop datanode

	# Clean HDFS --optional--
	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} rm -rf  ${cluster_base_dir}/hdfs/namenode
	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} rm -rf  ${cluster_base_dir}/hdfs/datanode

	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} mkdir -p "${cluster_base_dir}/hdfs/datanode"

	ssh -oStrictHostKeyChecking=no ubuntu@${wn} teaa/cluster/hadoop/bin/hdfs --daemon start datanode
done
