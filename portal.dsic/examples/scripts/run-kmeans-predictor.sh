#!/bin/bash

export PYTHONPATH="${HOME}/machine_learning_for_students"

#models_dir="models"
models_dir="models.pca"
pca="pca-"

results_dir="results4.pca.train"

clustering_id="1000"

hdfs dfs -cat data/uc13-${pca}train.csv \
    | python3 python/kmeans_uc13_predict.py  --codebook ${models_dir}/kmeans_model-uc13-${clustering_id}.pkl \
                                             --counter-pairs ${models_dir}/cluster-distribution-${clustering_id}.csv \
                                             --from-pca \
                                             --results-dir ${results_dir}
