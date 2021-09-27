#!/bin/bash

hdfs dfs -cat data/uc13-train.csv | python python/kmeans_uc13_predict.py --codebook models/kmeans_model-uc13-066.pkl --counter-pairs models/cluster-distribution-066.csv
