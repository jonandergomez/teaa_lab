#!/bin/bash 

dataset="digits"
results_dir="results/${dataset}"
summaries_dir="results/summaries"

for subset in "train" "test"
do
    for technique in "kde" "knn"
    do
        for filename in ${results_dir}/${technique}/${subset}/*.txt
        do
            set $(echo ${filename} | sed 's/-/ /g' | sed 's/_/ /g' | sed 's/\.txt/ /g' | awk '{ print $3, $5, $7 }')

            cb_size=$1
            pca=$2
            bw_or_K=$3

            f1_macro_avg=$(grep -H "^   macro avg" ${filename} | awk '{ print $(NF-1) }')
            running_time=$(grep -H "^running time in seconds:" ${filename} | awk '{ print $NF }')

            echo "${cb_size};${pca};${bw_or_K};${f1_macro_avg};${running_time}"

        done >/tmp/temp.${technique}.$$

        csv_file="/tmp/${dataset}-f1-macro-avg-evolution-${technique}-${subset}.csv"

        if [ "${technique}" = "kde" ]
        then
            (echo "codebook size;pca;band width;f1 macro avg;running time" ; cat /tmp/temp.${technique}.$$) >${csv_file}
        else
            (echo "codebook size;pca;K;f1 macro avg;running time" ; cat /tmp/temp.${technique}.$$) >${csv_file}
        fi

        rm -f /tmp/temp.${technique}.$$

        mkdir -p ${summaries_dir}
        mv ${csv_file} ${summaries_dir}
    done
done
