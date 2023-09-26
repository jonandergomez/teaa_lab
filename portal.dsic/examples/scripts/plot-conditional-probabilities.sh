#!/bin/bash

models_dir="models/digits/kmeans"
results_dir="results/digits/kmeans/train"

for filename in $(ls ${models_dir}/cluster-distribution-*.csv)
do
    echo $filename
    python3 python/see_conditional_probabilities.py --filename ${filename} --results-dir ${results_dir} --kmeans --save-figs
done

models_dir="models/digits/gmm"
results_dir="results/digits/gmm/train"

for filename in $(ls ${models_dir}/gmm-distribution-*.csv)
do
    echo $filename
    python3 python/see_conditional_probabilities.py --filename ${filename} --results-dir ${results_dir} --gmm --save-figs
done
