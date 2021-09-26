#!/bin/bash

for q in $(ls models/gmm-*.txt | cut -f2 -d'-' | cut -f1 -d'.')
do
    gmm_filename=$(printf "models/gmm-${q}.txt")
    if [ -f ${gmm_filename} ]
    then
        gmm_accumulators=$(printf "models/gmm-distribution-${q}.csv")
        if [ -f ${gmm_accumulators} ]
        then
            echo "we have this pair counters ${pair_counters}"
        else
            scripts/run-python.sh python/gmm_uc13.py  \
                                    --num-partitions 80 \
                                    --model ${gmm_filename} \
                                    --compute-confusion-matrix
        fi
    fi
done
