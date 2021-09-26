#!/bin/bash

num_clusters=0

case $(hostname) in
    teaa-*-cluster*)
        num_partitions=40
        ;;
    *)
        num_partitions=80
        ;;
esac

while [ $# -ge 1 ]
do
    case $1 in
        --num-clusters)
            num_clusters=$2
            shift
            ;;
        *)
            echo "unexpected option: $1"
            ;;
    esac
    shift
done



codebook=$(printf "models/kmeans_model-uc13-%03d.pkl" ${num_clusters})
if [ ! -f ${codebook} ]
then
    scripts/run-python.sh python/kmeans_uc13.py  --n-clusters ${num_clusters}
fi

counter_pairs=$(printf "models/cluster-distribution-%03d.csv" ${num_clusters})
if [ ! -f ${counter_pairs} ]
then
    scripts/run-python.sh python/kmeans_uc13_compute_confusion_matrix_spark.py --num-partitions 80  --codebook ${codebook}
fi

results_dir="results3.train"
results_file=$(printf "${results_dir}/classification-results-%03d.txt" ${num_clusters})
if [ ! -f ${results_file} ]
then
    scripts/run-python.sh python/kmeans_uc13_classifier.py  \
                                --dataset data/uc13-train.csv \
                                --results-dir ${results_dir} \
                                --num-partitions ${num_partitions} \
                                --codebook ${codebook} \
                                --counter-pairs ${counter_pairs}
fi

results_dir="results3.test"
results_file=$(printf "${results_dir}/classification-results-%03d.txt" ${num_clusters})
if [ ! -f ${results_file} ]
then
    scripts/run-python.sh python/kmeans_uc13_classifier.py  \
                                --dataset data/uc13-test.csv \
                                --results-dir ${results_dir} \
                                --num-partitions ${num_partitions} \
                                --codebook ${codebook} \
                                --counter-pairs ${counter_pairs}
fi
