#!/bin/bash 

dataset="uc13"

for subset in "train" "test"
do
    for technique in "kde" "knn"
    do
        for p in {01..24}
        do
            patient="chb${p}"
            results_dir="results.l3.${dataset}.${subset}/${technique}/${patient}"

            if [ -d ${results_dir} ]
            then
                echo ${results_dir}

                for filename in ${results_dir}/${technique}_*.txt
                do
                    set $(echo ${filename} | sed 's/-/ /g' | sed 's/_/ /g' | sed 's/\.txt/ /g' | awk '{ print $3, $5, $7, $8 }')

                    cb_size=$1
                    pca=$2
                    bw_or_K=$3
                    num_classes=$4

                    if [ "${pca}" = "0420" ]
                    then
                        pca="no_pca"
                    fi

                    f1_macro_avg=$(grep -H "^   macro avg" ${filename} | awk '{ print $(NF-1) }')
                    running_time=$(grep -H "^running time in seconds:" ${filename} | awk '{ print $NF }')

                    echo "${num_classes};${cb_size};${pca};${bw_or_K};${f1_macro_avg};${running_time}"

                done >/tmp/temp.${technique}.$$

                if [ "${technique}" = "kde" ]
                then
                    header="num classes;codebook size;pca;band width;f1 macro avg;running time"
                else
                    header="num classes;codebook size;pca;K;f1 macro avg;running time"
                fi

                csv_file="/tmp/${dataset}-${technique}-${patient}-${subset}-f1-macro-avg-evolution.csv"
                (echo "${header}" ; cat /tmp/temp.${technique}.$$) >${csv_file}
                rm -f /tmp/temp.${technique}.$$

                dest_dir="results.summary/l3/${dataset}/${technique}/${patient}"
                if [ -f ${csv_file} ]
                then
                    mkdir -p  ${dest_dir}
                    mv ${csv_file} ${dest_dir}
                fi
            fi
        done
    done
done
