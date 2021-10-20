#!/bin/bash

export PYTHONPATH="${HOME}/machine_learning_for_students"

master="spark://teaa-master-cluster.dsicv.upv.es:7077"

case $(hostname) in
    *teaa-*-cluster.dsicv.upv.es|*teaa-*-cluster)
        master="spark://teaa-master-cluster.dsicv.upv.es:7077"
        ;;
    *eibds01.mbda)
        master="spark://eibds01.mbda:7077"
        ;;
esac

program="$1"
shift

time \
spark-submit --master ${master} \
             --py-files ${PYTHONPATH}/mypythonlib.tgz,python/KernelClassifier.py,python/KNN_Classifier.py,python/BallTree.py \
             --deploy-mode client \
             ${program}  $*
