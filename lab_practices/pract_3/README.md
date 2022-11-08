# Lab practice 3

Lab practice 3 is about two use cases:

1. [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

2. Use Case 13 of the [DeepHealth EU project](https://deephealth-project.eu)
   based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0),
   and already presented in [Lab practice 1](../pract_1)


## Work to do in this lab practice related to Kernel Density Estimation (KDE)

1. Accessing to the cluster from a computer in the UPVNET domain or via Virtual Private Network (VPN) previously configured

    ```bash
        ssh -YC username@teaa-master-cluster.dsicv.upv.es
    ```

    please, replace `username` with yours.


1. **Inspect the code** [kde_mnist.py](../../portal.dsic/examples/python/kde_mnist.py)
   **to be used in this lab session for working with the**
   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

1. **Inspect the results already available** [here for train](../../portal.dsic/examples/results.l3.mnist.train/kde)
    and [here for test](../../portal.dsic/examples/results.l3.mnist.test/kde)

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on Kernel Density Estimation could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [kde_mnist.py](../../portal.dsic/examples/python/kde_mnist.py)
   **with different configuration options for** [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation):

    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KDE_, default value `0:100:200`
    - `--bandWidth`, _Colon separated list of the band width for the KDE classifier_, default value `0.1:0.2:0.5:1.0:2.0`
    - `--pcaComponents`, _Number of components of PCA, an integer > 1 or a float in the range [0,1[_, default value `37`

    ```bash
        teaa/examples/scripts/run-python-2.sh teaa/examples/python/kde_mnist.py \
                                --codebookSize 500 \
                                --bandWidth "0.5:1.0" \
                                --pcaComponents 0.90 \
                                --verbose 0
    ```

1. **Inspect the code** [kde_uc13_21x20.py](../../portal.dsic/examples/python/kde_uc13_21x20.py)
   **to be used in this lab session for working with the**
   dataset based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

1. **Inspect the results already available** [here for train](../../portal.dsic/examples/results.l3.uc13.train/kde)
    and [here for test](../../portal.dsic/examples/results.l3.uc13.test/kde)

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on Kernel Density Estimation could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [kde_uc13_21x20.py](../../portal.dsic/examples/python/kde_uc13_21x20.py)
   **with different configuration options for** [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation):

    - `patient`, _Patient identifier,_ e.g., `chb03`
    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KDE_, default value `0:100:200`
    - `--bandWidth`, _Colon separated list of the band width for the KDE classifier_, default value `0.1:0.2:0.5:1.0:2.0`
    - `--pcaComponents`, _Number of components of PCA, an integer > 1 or a float in the range [0,1[_, default value `37`
    - `--doBinaryClassification` or `--no-doBinaryClassification`
    - `--usingPCA` or `--no-usingPCA`
    
    ```bash
        teaa/examples/scripts/run-python-2.sh teaa/examples/python/kde_uc13_21x20.py chb10 \
                                        --codebookSize 200 \
                                        --bandWidth "0.5:1.0" \
                                        --usingPCA --doBinaryClassification \
                                        --verbose 0
    ```


## Work to do in this lab practice related to K-Nearest Neighbours (KNN)

1. **Inspect the code** [knn_mnist.py](../../portal.dsic/examples/python/knn_mnist.py)
   **to be used in this lab session for working with the**
   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

1. **Inspect the results already available** [here for train](../../portal.dsic/examples/results.l3.mnist.train/knn)
    and [here for test](../../portal.dsic/examples/results.l3.mnist.test/knn)

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on K-Nearest Neighbours could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [knn_mnist.py](../../portal.dsic/examples/python/knn_mnist.py)
   **with different configuration options for** [K-Nearest Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KNN_, default value `0:100:200`
    - `--K`, _Colon separated list of values for **K** the KNN classifier_, default value `3:5:7:9:11:13`
    - `--pcaComponents`, _Number of components of PCA, an integer > 1 or a float in the range [0,1[_, default value `37`

    ```bash
        teaa/examples/scripts/run-python-2.sh teaa/examples/python/knn_mnist.py \
                                --codebookSize 500 \
                                --K "5:7" \
                                --pcaComponents 0.90 \
                                --verbose 0
    ```

1. **Inspect the code** [knn_uc13_21x20.py](../../portal.dsic/examples/python/knn_uc13_21x20.py)
   **to be used in this lab session for working with the**
   dataset based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

1. **Inspect the results already available** [here for train](../../portal.dsic/examples/results.l3.uc13.train/knn)
    and [here for test](../../portal.dsic/examples/results.l3.uc13.test/knn)

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on K-Nearest Neighbours could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [knn_uc13_21x20.py](../../portal.dsic/examples/python/knn_uc13_21x20.py)
   **with different configuration options for** [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation):

    - `patient`, _Patient identifier,_ e.g., `chb03`
    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KNN_, default value `0:100:200`
    - `--K`, _Colon separated list of values for **K** the KNN classifier_, default value `3:5:7:9:11:13`
    - `--doBinaryClassification` or `--no-doBinaryClassification`
    - `--usingPCA` or `--no-usingPCA`
    
    ```bash
        teaa/examples/scripts/run-python-2.sh teaa/examples/python/knn_uc13_21x20.py chb10 \
                                        --codebookSize 200 \
                                        --K "5:7" \
                                        --usingPCA --doBinaryClassification \
                                        --verbose 0
    ```

## Final considerations

1. You can find a summary of the results for this lab practices [here](../../portal.dsic/examples/results.summary/l3)

    These summarized results will allow you to prepare plots where to observe the effect of different
    configuration parameters (a.k.a. hyper-parameters). In particular, `bandwith` for KDE classifers,
    `K` for KNN classifiers, and `codebook size` for both.

1. Report and preparation of the presentation

    As you can guess from the questions raised previously, you have to run some experiments (just a few ones)
    to check if some improvements can be obtained. Report it in the document you have to prepared for this
    lab practice, and if such findings are relevants, include them in the presentation.

    Report and presentation for this lab practice must focus on the effect of the above mentioned
    configuration hyperparameters for the two datasets used in this lab practice.
