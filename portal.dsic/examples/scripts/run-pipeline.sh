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

#models_dir="models"
models_dir="models.pca"
pca="pca-"
options="--from-pca"

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



codebook=$(printf "${models_dir}/kmeans_model-uc13-%03d.pkl" ${num_clusters})
if [ ! -f ${codebook} ]
then
    scripts/run-python.sh python/kmeans_uc13.py  --n-clusters ${num_clusters} ${options}
fi

counter_pairs=$(printf "${models_dir}/cluster-distribution-%03d.csv" ${num_clusters})
if [ ! -f ${counter_pairs} ]
then
    scripts/run-python.sh python/kmeans_uc13_compute_confusion_matrix_spark.py --num-partitions 80  --codebook ${codebook} ${options}
fi

results_dir="results3.train"
results_file=$(printf "${results_dir}/classification-results-%03d.txt" ${num_clusters})
if [ ! -f ${results_file} ]
then
    scripts/run-python.sh python/kmeans_uc13_classifier.py  \
                                --dataset data/uc13-${pca}train.csv \
                                --models-dir ${models_dir} \
                                --results-dir ${results_dir} \
                                --num-partitions ${num_partitions} \
                                --codebook ${codebook} \
                                --counter-pairs ${counter_pairs} ${options}
fi

results_dir="results3.test"
results_file=$(printf "${results_dir}/classification-results-%03d.txt" ${num_clusters})
if [ ! -f ${results_file} ]
then
    scripts/run-python.sh python/kmeans_uc13_classifier.py  \
                                --dataset data/uc13-${pca}test.csv \
                                --models-dir ${models_dir} \
                                --results-dir ${results_dir} \
                                --num-partitions ${num_partitions} \
                                --codebook ${codebook} \
                                --counter-pairs ${counter_pairs} ${options}
fi
