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
#data_filename_prefix="data/uc13"
data_filename_prefix="data/uc13-pca"

for q in $(ls ${models_dir}/gmm-*.txt | grep -v "0001" | cut -f2 -d'-' | cut -f1 -d'.')
do
    gmm_filename=$(printf "${models_dir}/gmm-${q}.txt")
    if [ -f ${gmm_filename} ]
    then
        counter_pairs=$(printf "${models_dir}/gmm-distribution-${q}.csv")
        if [ -f ${counter_pairs} ]
        then
            echo ${gmm_filename} ${counter_pairs}

            for subset in "train" "test"
            do
                results_dir="results3.${subset}"
                results_file=$(printf "${results_dir}/gmm-classification-results-${q}.txt")
                if [ ! -f ${results_file} ]
                then
                    scripts/run-python.sh python/gmm_uc13.py  \
                                        --num-partitions ${num_partitions} \
                                        --model ${gmm_filename} \
                                        --classify \
                                        --results-dir ${results_dir} \
                                        --reduce-labels \
                                        --models-dir ${models_dir} \
                                        --dataset ${data_filename_prefix}-${subset}.csv
                fi
            done
        fi
    fi
done
