#!/bin/bash

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

for q in $(ls ${models_dir}/kmeans_model-uc13* | cut -f3 -d'-' | cut -f1 -d'.')
do
    codebook=$(printf "${models_dir}/kmeans_model-uc13-${q}.pkl")
    if [ -f ${codebook} ]
    then
        #echo "we have this codebook ${codebook}"
        counter_pairs=$(printf "${models_dir}/cluster-distribution-${q}.csv")
        if [ -f ${counter_pairs} ]
        then
            results_file=$(printf "results3/classification-results-${q}.txt")
            if [ ! -f ${results_file} ]
            then
                scripts/run-python.sh python/kmeans_uc13_classifier.py  \
                                        --num-partitions ${num_partitions} \
                                        --codebook ${codebook} \
                                        --models-dir ${models_dir} \
                                        --counter-pairs ${counter_pairs} \
                                        --from-pca --no-reduce-labels
            fi
        fi
    fi
done
