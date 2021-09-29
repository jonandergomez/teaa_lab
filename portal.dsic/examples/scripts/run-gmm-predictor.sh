#!/bin/bash

#models_dir="models"
models_dir="models.pca"
pca="pca-"

hdfs dfs -cat data/uc13-${pca}train.csv | python python/gmm_uc13.py --model ${models_dir}/gmm-0031.txt --predict 
