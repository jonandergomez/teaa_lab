# Lab practice 2

Lab practice 2 is about two use cases:

1. [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

2. Use Case 13 of the [DeepHealth EU project](https://deephealth-project.eu)
   based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0),
   and already presented in [Lab practice 1](../pract_1)


## Work to do in this lab practice

1. Accessing to the cluster from a computer in the UPVNET domain or via Virtual Private Network (VPN) previously configured

   ```bash
   ssh -YC username@teaa-master-cluster.dsicv.upv.es
   ```

   please, replace `username` with yours.


2. Interaction with the Hadoop Distributed File System (HDFS)

   Once in the system, the first step is to see what we can find in the HDFS.
   The datasets to be used are in the directory of the `ubuntu` user, so we have
   to use the absolute path to see what it contains: `/user/ubuntu`.

   ![Here](figures/hdfs-01.png)

   It is relevant to see the directory were data is stored:

   ```bash
   hdfs dfs -ls /user/ubuntu/data
   ```

   and

   ```bash
   hdfs dfs -ls /user/ubuntu/data/uc13
   ```

3. Creation of local dirs in user home directory

   The directories are created from the Python code automatically, but if required,
   here some examples of how to create local directories in your home directory.

   ```bash
   cd ${HOME}

   mkdir results.l2.mnist.train
   mkdir results.l2.mnist.train/ert
   mkdir results.l2.mnist.train/rf
   mkdir results.l2.mnist.train/gbt

   mkdir -p results.l2.mnist.test/ert
   mkdir -p results.l2.mnist.test/rf
   mkdir -p results.l2.mnist.test/gbt
   ```

   And how to see the contents of the tree structure.

   ```bash
   cd ${HOME}

   tree results.l2.mnist.test
   ```

#4. Creation of the user directory in the HDFS
#
#   And as it will be necessary later, we have to create some directories in the HDFS.
#   In particular our own directory if it does not exist yet, and some another
#   directory as an example for discovering how directories are created.
#
#   ![Here](figures/hdfs-creating-user-dirs.png)
#
#   In particular, when using the Spark implentation of some training algorithms, the
#   models are stored in the HDFS, each user should create the directory corresponding
#   to the use case. In the case of working with the 
#   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)
#   we have to create `/user/jon/digits/models`, please, replace `jon` with your username.
#
#   ![Here](figures/hdfs-creating-models-dirs.png)


5. Inspect the code [rf_mnist.py](../../portal.dsic/examples/python/rf_mnist.py)
   to be used in this lab session for working with the
   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database),

6. Run the code [rf_mnist.py](../../portal.dsic/examples/python/rf_mnist.py)
   with different configuration options for [Random Forest](https://en.wikipedia.org/wiki/Random_forest):
    

    - `numTrees`: typical values range from 100 till 1000, but other values can be valid, use 100 in the first experiments
    - `maxDepth`: this depends on the dataset and the task, but take into account that a binary tree with depth 10 could have till 1024 leaf nodes (bins)
    - `pcaComponents`: this depends on the dataset and the task, but take into account that a binary tree with depth 10 could have till 1024 leaf nodes (bins)
    - `impurity` : depends on the ensemble type, basically, allowed values are `gini` and `entropy` **not changed, using the default one**

    Other command line options:

    - `baseDir`: base directory from which create others
    - `resultsDir`: directory relative to `baseDir` where to store results
    - `modelsDir`: directory relative to `baseDir` where to store models, **not used in this case**
    - `logDir`: directory relative to `baseDir` where to store logs of the execution, **not used in this case**

    Example of how to run the program:

   ```bash
   cd ${HOME}

   teaa/examples/scripts/run-python.sh teaa/examples/python/rf_mnist.py --numTrees 10 --maxDepth 7 --pcaComponents 40
   ```

7. What the program execution has created

   The models have not been stored in this example, and the results are stored in the local disk.
   Next instructions show you how to see the files created.

   ```bash
   cd ${HOME}

   ls -l results.l2.mnist.train

   ls -l results.l2.mnist.test
   ```

8. The results can be seen from the text file complementary to the figures

   ```bash
   cd ${HOME}

   cat results.l2.mnist.train/rf_00010_pca_0040_maxdepth_007.txt

   cat results.l2.mnist.test/rf_00010_pca_0040_maxdepth_007.txt
   ```

   The images can be downloaded to your computer and then visualized, anyway all the results are available
   in the repository exploring the following directories:

   - [results.l2.mnist.train](../../portal.dsic/examples/results.l2.mnist.train)
   - [results.l2.mnist.test](../../portal.dsic/examples/results.l2.mnist.test)

9. Do similar steps for the use case based on the
   [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

   ```bash
   cd ${HOME}

   teaa/examples/scripts/run-python.sh teaa/examples/python/rf_uc13.py chb03 --numTrees 50 --doBinaryClassification --usingPCA 
   ```


10. See where the models and the results have been stored:

   ```bash
   cd ${HOME}

   cat results.l2.uc13.train/rf/rf_chb03_00050_pca_02_classes.txt

   cat results.l2.uc13.test/rf/rf_chb03_00050_pca_02_classes.txt
   ```

   The images can be downloaded to your computer and then visualized, anyway all the results are available
   in the repository exploring the following directories:

   - [results.l2.uc13.train/rf](../../portal.dsic/examples/results.l2.uc13.train/rf)
   - [results.l2.uc13.test/rf](../../portal.dsic/examples/results.l2.uc13.test/rf)

11. Do the same using `extra-trees` instead of `random-forest`

    ***PENDING***


12. Do the same using `gradient-boosted-trees` instead of `random-forest` or `extra-trees` 

    ***PENDING***
