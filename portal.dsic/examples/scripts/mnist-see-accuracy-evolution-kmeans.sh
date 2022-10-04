#!/bin/bash 

grep -H accuracy results.digits.train/kmeans-classification-results-*.txt | cut -f4 -d'-' | sed 's/\.txt://g' | awk '{ print $1, $3 }' | sort -nk 1 >/tmp/temp.train.$$
grep -H accuracy results.digits.test/kmeans-classification-results-*.txt  | cut -f4 -d'-' | sed 's/\.txt://g' | awk '{ print $1, $3 }' | sort -nk 1 >/tmp/temp.test.$$


(echo "J train_acc test_acc" ; join /tmp/temp.train.$$ /tmp/temp.test.$$) | sed 's/ /;/g' >/tmp/mnist-kmeans-accuracy-evolution.csv


rm -f /tmp/temp.train.$$ /tmp/temp.test.$$


cat <<EOF
    
    See the file /tmp/mnist-kmeans-accuracy-evolution.csv or use it to represent a plot 
    of accuracy versus the size of the cluster obtained by using KMeans

EOF

