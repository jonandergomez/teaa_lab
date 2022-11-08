#!/bin/bash 

dataset="mnist"

for subset in "train" "test"
do
    for technique in "kde" "knn"
    do
        for filename in results.l3.${dataset}.${subset}/${technique}/${technique}_*.txt
        do
            set $(echo ${filename} | sed 's/-/ /g' | sed 's/_/ /g' | sed 's/\.txt/ /g' | awk '{ print $3, $5, $7 }')

            cb_size=$1
            pca=$2
            bw_or_K=$3

            accuracy=$(grep -H "^   macro avg" ${filename} | awk '{ print $(NF-1) }')
            running_time=$(grep -H "^running time in seconds:" ${filename} | awk '{ print $NF }')

            echo "${cb_size};${pca};${bw_or_K};${accuracy};${running_time}"

        done >/tmp/temp.${technique}.$$

        if [ "${technique}" = "kde" ]
        then
            (echo "codebook size;pca;band width;macro avg;running time" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${subset}-accuracy-evolution.csv
        else
            (echo "codebook size;pca;K;macro avg;running time" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${subset}-accuracy-evolution.csv
        fi

        rm -f /tmp/temp.${technique}.$$

        mkdir -p results.summary/l3/${dataset}
        mv /tmp/${dataset}-${technique}-${subset}-accuracy-evolution.csv results.summary/l3/${dataset}/
    done
done
