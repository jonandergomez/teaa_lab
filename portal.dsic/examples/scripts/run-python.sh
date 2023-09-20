#!/bin/bash

export PYTHONPATH="${HOME}/machine_learning_for_students"

export PYSPARK_PYTHON="/opt/anaconda3/bin/python"

master="spark://teaa-master-ubuntu22.dsicv.upv.es:7077"

case $(hostname) in
    *teaa-*-ubuntu22.dsicv.upv.es|*teaa-*-ubuntu22|*teaa-base-ubuntu22)
        master="spark://teaa-master-ubuntu22.dsicv.upv.es:7077"
        ;;
    *eibds01.mbda)
        master="spark://eibds01.mbda:7077"
        ;;
esac

program="$1"
shift

time \
spark-submit --master ${master} \
             --executor-memory 4G \
             --driver-memory 6G \
             --py-files ${PYTHONPATH}/mypythonlib.tgz \
             ${program}  $*

#             --archives ${PYTHONPATH}/mypythonlib.tgz \
