#!/bin/bash 

dataset="uc13"
results_dir="results/${dataset}"
target_dir="results/summaries"

for task in "binary-classification" "multi-class-classification"
do
for subset in "train" "test"
do
for technique in "kde" "knn"
do
for p in {01..24}
do
    patient="chb${p}"
    origin_dir="${results_dir}/${technique}/${patient}/${subset}"

    if [ -d ${origin_dir} ]
    then
        csv_file="/tmp/${dataset}-${patient}-f1-macro-avg-evolution-${technique}-${task}-${subset}.csv"
        count=$(ls ${origin_dir}/${technique}_kmeans_*_${task}.txt | wc -l)
        echo $p $count
        if [ ${count} -gt 3 ]
        then
            for filename in ${origin_dir}/${technique}_kmeans_*_${task}.txt
            do
                set $(echo ${filename} | sed 's/multi-class/multiclass/g' | sed 's/-/ /g' | sed 's/_/ /g' | sed 's/\.txt/ /g' | awk '{ print $3, $4, $6 }')

                cb_size=$1
                format=$2
                bw_or_K=$3

                f1_macro_avg=$(grep -H " macro avg " ${filename} | awk '{ print $(NF-1) }')
                running_time=$(grep -H "^running time in seconds:" ${filename} | awk '{ print $NF }')

                echo "${cb_size};${format};${bw_or_K};${f1_macro_avg};${running_time}"
            done >/tmp/temp.${technique}.$$

            if [ "${technique}" = "kde" ]
            then
                header="codebook size;format;pca;band width;f1 macro avg;running time"
            else
                header="codebook size;format;pca;K;f1 macro avg;running time"
            fi

            (echo "${header}" ; cat /tmp/temp.${technique}.$$) >${csv_file}
            rm -f /tmp/temp.${technique}.$$
            mkdir -p  ${target_dir}
            mv ${csv_file} ${target_dir}
        fi
    fi
done
done
done
done
