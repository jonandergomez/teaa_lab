# Lab practice 2

Lab practice 2 is about two use cases:

1. [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

2. Use Case 13 of the [DeepHealth EU project](https://deephealth-project.eu)
   based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0),
   and already presented in [Lab practice 1](../pract_1)


## Work to do in this lab practice

1. **Accessing to the cluster from a computer in the UPVNET domain or via Virtual Private Network (VPN) previously configured**

   ```bash
   ssh -YC username@teaa-master-cluster.dsicv.upv.es
   ```

   please, replace `username` with yours.


2. **Interaction with the Hadoop Distributed File System (HDFS)**

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

3. **Creation of local dirs in user home directory**

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

4. **Inspect the code** [rf_mnist.py](../../portal.dsic/examples/python/rf_mnist.py)
   **to be used in this lab session for working with the**
   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database),

5. **Run the code** [rf_mnist.py](../../portal.dsic/examples/python/rf_mnist.py)
   **with different configuration options for** [Random Forest](https://en.wikipedia.org/wiki/Random_forest):
    

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

   teaa/examples/scripts/run-python.sh teaa/examples/python/rf_mnist.py \
            --numTrees 10 --maxDepth 7 --pcaComponents 40
   ```

6. **What the program execution has created**

   The models have not been stored in this example, and the results are stored in the local disk.
   Next instructions show you how to see the files created.

   ```bash
   cd ${HOME}

   ls -l results.l2.mnist.train/rf

   ls -l results.l2.mnist.test/rf
   ```

7. **The results can be seen from the text file complementary to the figures**

   ```bash
   cd ${HOME}

   cat results.l2.mnist.train/rf/rf_00010_pca_0040_maxdepth_007.txt

   cat results.l2.mnist.test/rf/rf_00010_pca_0040_maxdepth_007.txt
   ```

   The images can be downloaded to your computer and then visualized, anyway all the results are available
   in the repository exploring the following directories:

   - [results.l2.mnist.train/rf](../../portal.dsic/examples/results.l2.mnist.train/rf)
   - [results.l2.mnist.test/rf](../../portal.dsic/examples/results.l2.mnist.test/rf)

8. **Do similar steps for the use case based on the**
   [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

   ```bash
   cd ${HOME}

   teaa/examples/scripts/run-python.sh teaa/examples/python/rf_uc13.py chb03 \
            --numTrees 50 --doBinaryClassification --usingPCA 
   ```


9. **See where the models and the results have been stored:**

   ```bash
   cd ${HOME}

   cat results.l2.uc13.train/rf/rf_chb03_00050_pca_02_classes.txt

   cat results.l2.uc13.test/rf/rf_chb03_00050_pca_02_classes.txt
   ```

   The images can be downloaded to your computer and then visualized, anyway all the results are available
   in the repository exploring the following directories:

   - [results.l2.uc13.train/rf/chb03](../../portal.dsic/examples/results.l2.uc13.train/rf/chb03)
   - [results.l2.uc13.test/rf/chb03](../../portal.dsic/examples/results.l2.uc13.test/rf/chb03)


10. **Do the same using `extra-trees` instead of `random-forest` for**
    [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database).
    **Inspect and run the code** [ert_mnist.py](../../portal.dsic/examples/python/ert_mnist.py)

    The command line options are the same used for Random Forest. See it above.
    In this case the `impurity` is used.

    ```bash
    cd ${HOME}

    teaa/examples/scripts/run-python.sh teaa/examples/python/ert_mnist.py \
            --numTrees 10 --maxDepth 7 --pcaComponents 40
    ```

    The results can be found with the following commands:

    ```bash
    cd ${HOME}

    ls -l results.l2.mnist.train/ert

    ls -l results.l2.mnist.test/ert
    ```

    And visualizing the text results with the following commands:

    ```bash
    cd ${HOME}

    cat results.l2.mnist.train/ert/ert_00010_pca_0040_maxdepth_007.txt

    cat results.l2.mnist.test/ert/ert_00010_pca_0040_maxdepth_007.txt
    ```

    The images can be downloaded to your computer and then visualized, anyway all the results are available
    in the repository exploring the following directories:

    - [results.l2.mnist.train/ert](../../portal.dsic/examples/results.l2.mnist.train/ert)
    - [results.l2.mnist.test/ert](../../portal.dsic/examples/results.l2.mnist.test/ert)


11. **Do the same using `extra-trees` instead of `random-forest` for**
    [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)
    **Inspect and run the code** [ert_uc13.py](../../portal.dsic/examples/python/ert_uc13.py)

    ```bash
    cd ${HOME}

    teaa/examples/scripts/run-python.sh teaa/examples/python/ert_uc13.py chb03 \
            --numTrees 50 --doBinaryClassification --usingPCA 
    ```

    The results can be found with the following commands:

    ```bash
    cd ${HOME}

    ls -l results.l2.uc13.train/ert/chb03

    ls -l results.l2.uc13.test/ert/chb03
    ```

    And visualizing the text results with the following commands:

    ```bash
    cd ${HOME}

    cat results.l2.uc13.train/ert/ert_00010_pca_0050_maxdepth_007.txt

    cat results.l2.uc13.test/ert/ert_00010_pca_0050_maxdepth_007.txt
    ```

    The images can be downloaded to your computer and then visualized, anyway all the results are available
    in the repository exploring the following directories:

    - [results.l2.uc13.train/ert](../../portal.dsic/examples/results.l2.uc13.train/ert)
    - [results.l2.uc13.test/ert](../../portal.dsic/examples/results.l2.uc13.test/ert)


12. **Do the same using `gradient-boosted-trees` instead of `random-forest` or `extra-trees` for**
    [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database).

	12.a. Using regressors then categorizing the predicted values.
		Inspect and run the code [gbt_mnist.py](../../portal.dsic/examples/python/gbt_mnist.py)

	The command line options are the same used for Random Forest and Extremely Randomized Trees. See it above.
	In this case the `impurity` is used but fixed to `variance`.

	```bash
	cd ${HOME}

	teaa/examples/scripts/run-python.sh teaa/examples/python/gbt_mnist.py \
            --numTrees 10 --maxDepth 7 --pcaComponents 40
	```

      The results can be found with the following commands:

      ```bash
      cd ${HOME}

      ls -l results.l2.mnist.train/gbt

      ls -l results.l2.mnist.test/gbt
      ```

      And visualizing the text results with the following commands:

      ```bash
      cd ${HOME}

      cat results.l2.mnist.train/gbt/gbt_00010_pca_0040_maxdepth_007.txt

      cat results.l2.mnist.test/gbt/gbt_00010_pca_0040_maxdepth_007.txt
      ```

      The images can be downloaded to your computer and then visualized, anyway all the results are available
      in the repository exploring the following directories:

      - [results.l2.mnist.train/gbt](../../portal.dsic/examples/results.l2.mnist.train/gbt)
      - [results.l2.mnist.test/gbt](../../portal.dsic/examples/results.l2.mnist.test/gbt)

  
	12.b. Using a cascade of binary classifiers.
		Inspect and run the code [gbt_mnist_binary_trees.py](../../portal.dsic/examples/python/gbt_mnist_binary_trees.py)

	The command line options are the same used in previous examples. See it above.

    ```bash
    cd ${HOME}

    teaa/examples/scripts/run-python.sh teaa/examples/python/gbt_mnist_binary_trees.py \
        --numTrees 10 --maxDepth 7 --pcaComponents 40
    ```

    The results can be found with the following commands:

    ```bash
    cd ${HOME}

    ls -l results.l2b.mnist.train/gbt

    ls -l results.l2b.mnist.test/gbt
    ```

    And visualizing the text results with the following commands:

    ```bash
    cd ${HOME}

    cat results.l2b.mnist.train/gbt/gbt_00010_pca_0040_maxdepth_007.txt

    cat results.l2b.mnist.test/gbt/gbt_00010_pca_0040_maxdepth_007.txt
    ```

    The images can be downloaded to your computer and then visualized, anyway all the results are available
    in the repository exploring the following directories:

    - [results.l2b.mnist.train/gbt](../../portal.dsic/examples/results.l2b.mnist.train/gbt)
    - [results.l2b.mnist.test/gbt](../../portal.dsic/examples/results.l2b.mnist.test/gbt)


13.  ***PENDING***
