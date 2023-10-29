#!/bin/bash 

dataset="uc13"
results_dir="results/${dataset}/ensembles"
target_dir="results/summaries"

for task in "binary-classification" "multi-class-classification" "regression"
do
for subset in "train" "test"
do
for technique in "rf" "ert" "gbt"
do
for p in {01..24}
do
    patient="chb${p}"
    origin_dir="${results_dir}/${patient}/${technique}/${subset}"

    if [ -d ${origin_dir} ]
    then
        csv_file="/tmp/${dataset}-${patient}-f1-macro-avg-evolution-${technique}-${task}-${subset}.csv"
        count=$(ls ${origin_dir}/${technique}-*-${task}-*.txt | wc -l)
        if [ ${count} -gt 3 ]
        then
            grep -H "^   macro avg" ${origin_dir}/${technique}-*-${task}-*.txt \
                | sed 's/multi-class/multiclass/g' \
                | sed 's/-/ /g' \
                | sed 's/\.txt/ /g' \
                | awk '{ patient=$2; format=$3; nt=$6; md=$7;
                           printf("%s;%s;%s;%s;%s\n", patient, format, nt, md, $(NF-1)); }' >/tmp/temp.${technique}.$$

                (echo "patient;data_format;num_trees;max_depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >${csv_file}
                rm -f /tmp/temp.${technique}.$$
            mkdir -p  ${target_dir}
            mv ${csv_file} ${target_dir}
        fi
    fi
done
done
done
done
