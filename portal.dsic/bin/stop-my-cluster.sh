#!/bin/bash

${SPARK_HOME}/sbin/stop-all.sh

# ${HADOOP_HOME}/bin/mapred historyserver stop

${HADOOP_HOME}/bin/hdfs --daemon stop httpfs

${HADOOP_HOME}/sbin/stop-all.sh

