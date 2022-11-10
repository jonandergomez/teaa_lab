#!/bin/bash 

dataset="uc13"

for subset in "train" "test"
do
    for technique in "rf" "ert" "gbt"
    do
        for p in {01..24}
        do
            patient="chb${p}"
            results_dir="results.l2.${dataset}.${subset}/${technique}/${patient}"

            if [ -d ${results_dir} ]
            then
                for num_classes in 02 10
                do
                    case ${technique} in
                        rf)
                            grep -H "^   macro avg" ${results_dir}/${technique}_*_${num_classes}_classes.txt \
                                | sed 's/no_pca/nopca/g' \
                                | sed 's/-/ /g' \
                                | sed 's/_/ /g' \
                                | sed 's/nopca/no_pca/g' \
                                | sed 's/\.txt/ /g' \
                                | awk '{ print $3, $4, $(NF-1) }' \
                                | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$

                            (echo "num trees;pca;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv
                            rm -f /tmp/temp.${technique}.$$
                            ;;
                        
                        ert)
                            grep -H "^   macro avg" ${results_dir}/${technique}_*_${num_classes}_classes.txt \
                                | sed 's/no_pca/nopca/g' \
                                | sed 's/-/ /g' \
                                | sed 's/_/ /g' \
                                | sed 's/nopca/no_pca/g' \
                                | sed 's/\.txt/ /g' \
                                | awk '{ print $3, $6, $5, $(NF-1) }' \
                                | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$

                            (echo "num trees;pca;max_depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv
                            rm -f /tmp/temp.${technique}.$$
                            ;;
                        
                        gbt)
                            if [ ${num_classes} = "02" ]
                            then
                                grep -H "^   macro avg" ${results_dir}/${technique}_*_${num_classes}_classes.txt \
                                    | sed 's/no_pca/nopca/g' \
                                    | sed 's/-/ /g' \
                                    | sed 's/_/ /g' \
                                    | sed 's/nopca/no_pca/g' \
                                    | sed 's/\.txt/ /g' \
                                    | awk '{ print $3, $5, $4, $(NF-1) }' \
                                    | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$

                                (echo "num trees;pca;max_depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv
                                rm -f /tmp/temp.${technique}.$$
                            fi
                            ;;
                   esac 




                    csv_file="/tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv"
                    dest_dir="results.summary/l2/${dataset}/${technique}/${patient}"
                    if [ -f ${csv_file} ]
                    then
                        mkdir -p  ${dest_dir}
                        mv /tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv ${dest_dir}
                    fi
                done
            fi
        done
    done
done

#results.l2.uc13.test/rf/chb24/rf_chb24_00700_no_pca_10_classes.txt
#results.l2.uc13.test/rf/chb24/rf_chb24_00700_pca_02_classes.txt

#results.l2.uc13.test/ert/chb24/ert_chb24_01000_maxdepth_009_pca_10_classes.txt
#results.l2.uc13.test/ert/chb24/ert_chb24_01000_maxdepth_013_no_pca_02_classes.txt

#results.l2.uc13.test/gbt/chb24/gbt_chb24_00050_005_no_pca_02_classes.txt
#results.l2.uc13.test/gbt/chb24/gbt_chb24_00050_005_pca_02_classes.txt

#results.l2c.uc13.test/gbt/chb24/gbt_chb24_00020_009_pca.txt
#results.l2c.uc13.test/gbt/chb24/gbt_chb24_00020_011_no_pca.txt

for subset in "train" "test"
do
    technique="gbt"
    for p in {01..24}
    do
        patient="chb${p}"
        results_dir="results.l2c.${dataset}.${subset}/${technique}/${patient}"

        if [ -d ${results_dir} ]
        then
            num_classes="04"
            grep -H "^   macro avg" ${results_dir}/${technique}_*.txt \
                    | sed 's/no_pca/nopca/g' \
                    | sed 's/-/ /g' \
                    | sed 's/_/ /g' \
                    | sed 's/nopca/no_pca/g' \
                    | sed 's/\.txt/ /g' \
                    | awk '{ print $3, $5, $4, $(NF-1) }' \
                    | awk '{ for(i=1; i <= NF; i++) printf("%s;", $i); printf("\n"); }' >/tmp/temp.${technique}.$$

            (echo "num trees;pca;max_depth;f1 macro avg" ; cat /tmp/temp.${technique}.$$) >/tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv
            rm -f /tmp/temp.${technique}.$$
        fi


        csv_file="/tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv"
        dest_dir="results.summary/l2c/${dataset}/${technique}/${patient}"
        if [ -f ${csv_file} ]
        then
            mkdir -p  ${dest_dir}
            mv /tmp/${dataset}-${technique}-${patient}-${num_classes}-classes-${subset}-f1-macro-avg-evolution.csv ${dest_dir}
        fi
    done
done
