#!/bin/bash

program="$1"
shift

time \
spark-submit --master spark://teaa-master-cluster:7077 \
             --py-files ${PYTHONPATH}/mypythonlib.tgz \
             ${program}  $*
