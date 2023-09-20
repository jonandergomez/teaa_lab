#!/bin/bash

workers=$(cat ${SPARK_HOME}/conf/workers | grep -v "^#" | grep -v "^$" | awk '{ print $1 }')
cluster_base_dir="${HOME}/teaa/disk"


for wn in ${workers}
do
	echo "${wn}"

	# Clean HDFS --optional--
	ssh -oStrictHostKeyChecking=no ${wn} rm -rf  ${cluster_base_dir}/hdfs/namenode
	ssh -oStrictHostKeyChecking=no ${wn} rm -rf  ${cluster_base_dir}/hdfs/datanode

	ssh -oStrictHostKeyChecking=no ${wn} mkdir -p "${cluster_base_dir}/hdfs/datanode"
done

# Do it in the master, i.e., the namemode

rm -rf   ${cluster_base_dir}/hdfs/namenode
rm -rf   ${cluster_base_dir}/hdfs/datanode
mkdir -p ${cluster_base_dir}/hdfs/namenode
mkdir -p ${cluster_base_dir}/hdfs/datanode
