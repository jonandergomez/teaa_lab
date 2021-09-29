#!/bin/bash

#models_dir="models"
models_dir="models.pca"

for q in $(ls ${models_dir}/kmeans_model-uc13* | cut -f3 -d'-' | cut -f1 -d'.')
do
    codebook=$(printf "${models_dir}/kmeans_model-uc13-${q}.pkl")
    if [ -f ${codebook} ]
    then
        #echo "we have this codebook ${codebook}"
        pair_counters=$(printf "${models_dir}/cluster-distribution-${q}.csv")
        if [ -f ${pair_counters} ]
        then
            echo "we have this pair counters ${pair_counters}"
        else
            scripts/run-python.sh python/kmeans_uc13_compute_confusion_matrix_spark.py  \
                                    --num-partitions 80 \
                                    --codebook ${codebook} \
                                    --models_dir ${models_dir} \
                                    --from-pca --no-reduce-labels
        fi
    fi
done
