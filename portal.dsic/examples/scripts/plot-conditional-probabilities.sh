#!/bin/bash

#models_dir="models"
#models_dir="models.pca"
models_dir="models.digits"

#results_dir="results4.pca.train"
results_dir="results.digits.train"

for filename in $(ls ${models_dir}/cluster-distribution-*.csv)
do
    echo $filename
    python3 python/see_conditional_probabilities.py --filename ${filename} --results-dir ${results_dir} --kmeans --save-figs
done

for filename in $(ls ${models_dir}/gmm-distribution-*.csv)
do
    echo $filename
    python3 python/see_conditional_probabilities.py --filename ${filename} --results-dir ${results_dir} --gmm --save-figs
done
