#!/bin/bash

models_dir="models.digits"

for q in $(ls ${models_dir}/gmm-*.txt | grep -v "0001" | cut -f2 -d'-' | cut -f1 -d'.')
do
    gmm_filename=$(printf "${models_dir}/gmm-${q}.txt")
    if [ -f ${gmm_filename} ]
    then
        gmm_accumulators=$(printf "${models_dir}/gmm-distribution-${q}.csv")
        if [ -f ${gmm_accumulators} ]
        then
            echo "we have this pair counters ${pair_counters}"
        else
            scripts/run-python.sh python/gmm_mnist.py  \
                                    --num-partitions 80 \
                                    --model ${gmm_filename} \
                                    --compute-confusion-matrix \
                                    --models-dir ${models_dir} 
        fi
    fi
done
