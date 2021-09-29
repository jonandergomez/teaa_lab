#!/bin/bash

#models_dir="models"
models_dir="models.pca"
pca="pca-"

hdfs dfs -cat data/uc13-${pca}train.csv | python python/kmeans_uc13_predict.py --codebook ${models_dir}/kmeans_model-uc13-066.pkl --counter-pairs ${models_dir}/cluster-distribution-066.csv
