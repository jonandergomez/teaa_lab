#!/bin/bash

export PYTHONPATH="${HOME}/machine_learning_for_students"

#models_dir="models"
models_dir="models.pca"
pca="pca-"

results_dir="results4.pca.train"

#gmm_ids="0032"
gmm_ids="$(ls models.pca/gmm-distribution-0*.csv | cut -f3 -d'-' | cut -f1 -d'.')"

for gmm_id in ${gmm_ids}
do
    hdfs dfs -cat data/uc13-${pca}train.csv | python3 python/gmm_uc13.py --model ${models_dir}/gmm-${gmm_id}.txt --predict  --from-pca --results-dir ${results_dir}
done
