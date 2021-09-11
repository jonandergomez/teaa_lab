#!/bin/bash

${HADOOP_HOME}/sbin/start-all.sh

${HADOOP_HOME}/bin/hdfs --daemon start httpfs

${SPARK_HOME}/sbin/start-all.sh

# ${HADOOP_HOME}/bin/mapred historyserver start  &
