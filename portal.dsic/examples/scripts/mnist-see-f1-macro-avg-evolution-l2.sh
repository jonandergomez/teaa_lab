#!/bin/bash 

dataset="mnist"

for subset in "train" "test"
do
    for technique in "rf" "ert" "gbt"
    do

        grep -H "^   macro avg" results.l2.${dataset}.${subset}/${technique}/${technique}_*.txt \
            | sed 's/-/ /g' \
            | sed 's/_/ /g' \
            | sed 's/\.txt/ /g' \
            | awk '{ print $2, $4, $6, $(NF-1) }' \
            | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$


        csv_file="/tmp/${dataset}-f1-macro-avg-evolution-${technique}-${subset}.csv"
        (echo "num trees;pca;max depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >${csv_file}

        rm -f /tmp/temp.${technique}.$$

        mkdir -p results.summary/l2/${dataset}
        mv ${csv_file} results.summary/l2/${dataset}/
   done
done

for subset in "train" "test"
do
    for technique in "gbt"
    do

        grep -H "^   macro avg" results.l2b.${dataset}.${subset}/${technique}/${technique}_*.txt \
            | sed 's/-/ /g' \
            | sed 's/_/ /g' \
            | sed 's/\.txt/ /g' \
            | awk '{ print $2, $4, $6, $(NF-1) }' \
            | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$


        csv_file="/tmp/${dataset}-f1-macro-avg-evolution-${technique}-${subset}.csv"
        (echo "num trees;pca;max depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >${csv_file}

        rm -f /tmp/temp.${technique}.$$

        mkdir -p results.summary/l2b/${dataset}
        mv ${csv_file} results.summary/l2b/${dataset}/
   done
done
