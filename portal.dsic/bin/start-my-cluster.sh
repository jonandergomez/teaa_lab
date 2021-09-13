#!/bin/bash


mkdir -p /tmp/spark-events
chgrp -R hdfs /tmp/spark-events

${HADOOP_HOME}/sbin/start-all.sh

${HADOOP_HOME}/bin/hdfs --daemon start httpfs

${SPARK_HOME}/sbin/start-all.sh

# ${HADOOP_HOME}/bin/mapred historyserver start  &
