#!/bin/bash


cluster_base_dir="teaa/disk"

for n in $*
do
    wn="teaa-worker${n}-cluster.dsicv.upv.es"
	echo "${wn}"

	# Clean HDFS --optional--
	ssh -oStrictHostKeyChecking=no ubuntu@${wn} rm -rf  ${cluster_base_dir}/hdfs/namenode
	ssh -oStrictHostKeyChecking=no ubuntu@${wn} rm -rf  ${cluster_base_dir}/hdfs/datanode

	ssh -oStrictHostKeyChecking=no ubuntu@${wn} mkdir -p "${cluster_base_dir}/hdfs/datanode"

	#ssh -oStrictHostKeyChecking=no ubuntu@${wn} hdfs --daemon start datanode
done
