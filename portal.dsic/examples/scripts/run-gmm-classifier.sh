#!/bin/bash

case $(hostname) in
    teaa-*-cluster*)
        num_partitions=40
        ;;
    *)
        num_partitions=80
        ;;
esac

for q in $(ls models/gmm-*.txt | cut -f2 -d'-' | cut -f1 -d'.')
do
    gmm_filename=$(printf "models/gmm-${q}.txt")
    if [ -f ${gmm_filename} ]
    then
        counter_pairs=$(printf "models/gmm-distribution-${q}.csv")
        if [ -f ${counter_pairs} ]
        then
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
                                        --dataset data/uc13-${subset}.csv
                fi
            done
        fi
    fi
done
