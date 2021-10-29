import os
import sys
import random

generate_for_mnist = False
generate_for_uc13_1 = True
generate_for_uc13_21x20 = False



ensemble_types_a = ['random-forest', 'extra-trees']
ensemble_types_b = ['gradient-boosted-trees']
num_trees_a = [i for i in range(100, 300, 50)] + [i for i in range(300, 1001, 100)]
num_trees_b = [i for i in range(10, 30, 5)]
impurities = ['gini'] #, 'entropy']
max_depths_a = [i for i in range(3, 10 + 1)]
max_depths_b = [i for i in range(3, 7 + 1)]

if generate_for_mnist:

    command_line = 'scripts/run-python-2.sh python/tree_ensembles_mnist.py --num-partitions 80 --ensemble-type %s --num-trees %d --num-iterations %d --impurity %s --max-depth %d --train --classify'

    print("#!/bin/bash")

    for et in ensemble_types_a:
        for nt in num_trees_a:
            for md in max_depths_a:
                for imp in impurities:
                    print(command_line % (et, nt, nt, imp, md))

    for et in ensemble_types_b:
        for nt in num_trees_b:
            for md in max_depths_b:
                for imp in impurities:
                    print(command_line % (et, nt, nt, imp, md))
    sys.exit(0)

if generate_for_uc13_1:

    command_line_train = 'scripts/run-python-2.sh python/tree_ensembles_uc13.py --num-partitions 80 --ensemble-type %s --num-trees %d --num-iterations %d --impurity %s --max-depth %d --subset train --train --classify --reduce-labels'
    command_line_test  = 'scripts/run-python-2.sh python/tree_ensembles_uc13.py --num-partitions 80 --ensemble-type %s --num-trees %d --num-iterations %d --impurity %s --max-depth %d --subset test          --classify --reduce-labels'

    print("#!/bin/bash")

    for et in ensemble_types_a:
        for nt in num_trees_a:
            for md in max_depths_a:
                for imp in impurities:
                    #print(command_line_train % (et, nt, nt, imp, md))
                    print(command_line_test % (et, nt, nt, imp, md))

    for et in ensemble_types_b:
        for nt in num_trees_b:
            for md in max_depths_b:
                for imp in impurities:
                    #print(command_line_train % (et, nt, nt, imp, md))
                    print(command_line_test % (et, nt, nt, imp, md))
    sys.exit(0)



if generate_for_uc13_21x20:

    patients = ['chb03', 'chb07', 'chb10']

    command_line = 'scripts/run-python-2.sh python/tree_ensembles_uc13_21x20.py --num-partitions 80 --patient %s --ensemble-type %s --num-trees %d --num-iterations %d --impurity %s --max-depth %d --train --classify --reduce-labels'

    print("#!/bin/bash")

    for patient in patients:
        for et in ensemble_types_a:
            for nt in num_trees_a:
                for md in max_depths_a:
                    for imp in impurities:
                        print(command_line % (patient, et, nt, nt, imp, md))

    for patient in patients:
        for et in ensemble_types_b:
            for nt in num_trees_b:
                for md in max_depths_b:
                    for imp in impurities:
                        print(command_line % (patient, et, nt, nt, imp, md))
    sys.exit(0)
