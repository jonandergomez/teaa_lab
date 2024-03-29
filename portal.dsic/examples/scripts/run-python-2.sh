#!/bin/bash

export PYTHONPATH="${HOME}/machine_learning_for_students"
export PYSPARK_PYTHON="/opt/anaconda3/bin/python"

master="spark://teaa-master-ubuntu22.dsicv.upv.es:7077"
python_dir="/home/ubuntu/teaa/examples/python"

case $(hostname) in
    *teaa-*-ubuntu22.dsicv.upv.es|*teaa-*-ubuntu22)
        master="spark://teaa-master-ubuntu22.dsicv.upv.es:7077"
        python_dir="/home/ubuntu/teaa/examples/python"
        ;;
    *eibds01.mbda)
        master="spark://eibds01.mbda:7077"
        python_dir="${HOME}/cluster/examples/python"
        ;;
esac

deploy_mode="client"

case $0 in
    *launch-python*.sh)
        deploy_mode="cluster"
        ;;
    *)
        deploy_mode="client"
        ;;
esac

program="$1"
shift


time \
spark-submit --master ${master} \
             --executor-memory 4G \
             --driver-memory 6G \
             --py-files ${PYTHONPATH}/mypythonlib.tgz,${python_dir}/KernelClassifier.py,${python_dir}/KNN_Classifier.py,${python_dir}/BallTree.py,${python_dir}/eeg_load_data.py \
             --deploy-mode ${deploy_mode} \
             ${program}  $*
