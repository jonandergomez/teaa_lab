#!/bin/bash

hdfs dfs -cat data/uc13-train.csv | python python/gmm_uc13.py --model models/gmm-0031.txt --predict 
