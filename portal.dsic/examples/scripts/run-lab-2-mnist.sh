#!/bin/bash

num_partitions=60

# RANDOM FOREST

for num_trees in 100 150 200 250 300 400 500 600 700 800 900 100
do
    for max_depth in {3..10}
    do
        echo scripts/run-python.sh python/tree_ensembles_mnist.py \
            --num-partitions ${num_partitions} \
            --ensemble-type random-forest \
            --num-trees ${num_trees} \
            --num-iterations ${num_trees} \
            --impurity gini \
            --max-depth ${max_depth} \
            --train --classify
    done
done

# EXTREMELY RANDOMIZED TREES

for num_trees in 100 150 200 250 300 400 500 600 700 800 900 100
do
    for max_depth in {3..10}
    do
        echo scripts/run-python.sh python/tree_ensembles_mnist.py \
            --num-partitions ${num_partitions} \
            --ensemble-type extra-trees \
            --num-trees ${num_trees} \
            --num-iterations ${num_trees} \
            --impurity gini \
            --max-depth ${max_depth} \
            --train --classify
    done
done

# GRADIENT BOOSTED TREES

for num_trees in 10 15 20 30 50 100
do
    for max_depth in {3..7}
    do
        echo scripts/run-python.sh python/tree_ensembles_mnist.py \
            --num-partitions ${num_partitions} \
            --ensemble-type gradient-boosted-trees \
            --num-trees ${num_trees} \
            --num-iterations ${num_trees} \
            --impurity gini \
            --max-depth ${max_depth} \
            --train --classify
    done
done
