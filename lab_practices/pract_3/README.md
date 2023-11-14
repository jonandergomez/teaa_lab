# Lab practice 3

Lab practice 3 is about two use cases:

1. [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

2. [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)
   already presented in [Lab practice 1](../pract_1)


## Work to do in this lab practice related to Kernel Density Estimation (KDE)

1. Accessing to the cluster from a computer in the UPVNET domain or via Virtual Private Network (VPN) previously configured

    ```bash
        ssh -YC username@teaa-master-ubuntu22.dsicv.upv.es
    ```

    please, replace `username` with yours.


1. **Inspect the code** [kde_mnist.py](../../portal.dsic/examples/python/kde_mnist.py)
   **to be used in this lab session for working with the**
   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

1. **Inspect the results already available** [here the ones obtained with the training subset](../../portal.dsic/examples/results/digits/kde/train)
   and [here the ones obtained with the testing subset](../../portal.dsic/examples/results/digits/kde/test)

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on Kernel Density Estimation could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [kde_mnist.py](../../portal.dsic/examples/python/kde_mnist.py)
   **with different configuration options for** [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation):

    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KDE_, default value `0:100:200`
    - `--bandWidth`, _Colon separated list of the band width for the KDE classifier_, default value `0.1:0.2:0.5:1.0:2.0`
    - `--pcaComponents`, _Number of components of PCA, an integer > 1 or a float in the range [0,1[_, default value `37`

    Example:
    ```bash
        cd  ${HOME}

        teaa/examples/scripts/run-python-2.sh teaa/examples/python/kde_mnist.py \
                                --codebookSize 500 \
                                --bandWidth "0.5:1.0" \
                                --pcaComponents 0.90 \
                                --verbose 0
    ```

   You can see the [script used to run all the experiments](../../portal.dsic/examples/scripts/run-mnist-kde-grid.sh)
   whose results are already available in this repository as indicated above.

1. **Inspect the code** [kde_uc13_21x14.py](../../portal.dsic/examples/python/kde_uc13_21x14.py)
   **to be used in this lab session for working with the**
   dataset based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

1. **Inspect the results already available** [here the ones obtained with the training subset](../../portal.dsic/examples/results/uc13/kde/chb03/train)
   and [here the ones obtained with the testing subset](../../portal.dsic/examples/results/uc13/kde/chb03/test).
   What you can see in the previous links corresponds to patient `chb03`, changhe the patient id accordingly to see the results of others.

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on Kernel Density Estimation could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [kde_uc13_21x14.py](../../portal.dsic/examples/python/kde_uc13_21x14.py)
   **with different configuration options for** [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation):

    - `patient`, _Patient identifier,_ e.g., `chb03`
    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KDE_, default value `0:100:200`
    - `--bandWidth`, _Colon separated list of the band width for the KDE classifier_, default value `0.1:0.2:0.5:1.0:2.0`
    - `--pcaComponents`, _Number of components of PCA, an integer > 1 or a float in the range [0,1[_, default value `37`
    - `--doBinaryClassification` or `--no-doBinaryClassification`
    - `--format` _Format ID_, valid values are `pca136` or `21x14`, default value is `21x14`
    
    Example:
    ```bash
        cd  ${HOME}

        teaa/examples/scripts/run-python-2.sh teaa/examples/python/kde_uc13_21x14.py chb10 \
                                        --codebookSize 200 \
                                        --bandWidth "0.5:1.0" \
                                        --format 21x14 \
                                        --doBinaryClassification \
                                        --verbose 0

    ```

   You can see the [script used to run all the experiments](../../portal.dsic/examples/scripts/run-uc13-kde-grid.sh)
   whose results are already available in this repository as indicated above.


## Work to do in this lab practice related to K-Nearest Neighbours (KNN)

1. **Inspect the code** [knn_mnist.py](../../portal.dsic/examples/python/knn_mnist.py)
   **to be used in this lab session for working with the**
   [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

1. **Inspect the results already available** [here the ones obtained with the training subset](../../portal.dsic/examples/results/digits/knn/train)
   and [here the ones obtained with the testing subset](../../portal.dsic/examples/results/digits/knn/test)

    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on K-Nearest Neighbours could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [knn_mnist.py](../../portal.dsic/examples/python/knn_mnist.py)
   **with different configuration options for** [K-Nearest Neighbours](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KNN_, default value `0:100:200`
    - `--K`, _Colon separated list of values for **K** the KNN classifier_, default value `3:5:7:9:11:13`
    - `--pcaComponents`, _Number of components of PCA, an integer > 1 or a float in the range [0,1[_, default value `37`

    Example:
    ```bash
        cd  ${HOME}

        teaa/examples/scripts/run-python-2.sh teaa/examples/python/knn_mnist.py \
                                --codebookSize 500 \
                                --K "5:7" \
                                --pcaComponents 0.90 \
                                --verbose 0
    ```

   You can see the [script used to run all the experiments](../../portal.dsic/examples/scripts/run-mnist-knn-grid.sh)
   whose results are already available in this repository as indicated above.

1. **Inspect the code** [knn_uc13_21x14.py](../../portal.dsic/examples/python/knn_uc13_21x14.py)
   **to be used in this lab session for working with the**
   dataset based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

1. **Inspect the results already available** [here the ones obtained with the training subset](../../portal.dsic/examples/results/uc13/knn/chb03/train)
   and [here the ones obtained with the testing subset](../../portal.dsic/examples/results/uc13/knn/chb03/test).
   What you can see in the previous links corresponds to patient `chb03`, changhe the patient id accordingly to see the results of others.


    Do you think some non-used combinations of the hyper-parameters for the classifier based
    on K-Nearest Neighbours could lead to better results?

    Do you observe overfitting?
    
    In the affirmative case, what do you think about the overfitting when using this machine learning technique?

1. **Run the code** [knn_uc13_21x14.py](../../portal.dsic/examples/python/knn_uc13_21x14.py)
   **with different configuration options for** [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation):

    - `patient`, _Patient identifier,_ e.g., `chb03`
    - `--codebookSize`, _Colon separated list of the codebook sizes to apply kmeans before KNN_, default value `0:100:200`
    - `--K`, _Colon separated list of values for **K** the KNN classifier_, default value `3:5:7:9:11:13`
    - `--doBinaryClassification` or `--no-doBinaryClassification`
    - `--format` _Format ID_, valid values are `pca136` or `21x14`, default value is `21x14`
    
    Example:
    ```bash
        cd ${HOME}

        teaa/examples/scripts/run-python-2.sh teaa/examples/python/knn_uc13_21x14.py chb10 \
                                        --codebookSize 200 \
                                        --K "5:7" \
                                        --format "21x14" \
                                        --doBinaryClassification \
                                        --verbose 0
    ```

   You can see the [script used to run all the experiments](../../portal.dsic/examples/scripts/run-uc13-knn-grid.sh)
   whose results are already available in this repository as indicated above.

## Final considerations

1. You can find a summary of the results for this lab practice [here](../../portal.dsic/examples/results/summaries/)

    These summarized results will allow you to prepare plots where to observe the effect of different
    configuration parameters (a.k.a. hyper-parameters). In particular, `bandwith` for KDE classifers,
    `K` for KNN classifiers, and `codebook size` for both.

1. The summary of results for each kind of experiment includes the running times of doing inference with both training
   and testing subsets. 

   Is it worth to do K-Means in order to reduce the size of the training subset used in these memory-based techniques
   to mitigate the problem of the time required to classify each sample?

   For which of both techniques is the reduction of inference time more relevant?

1. Report and preparation of the presentation

    As you can guess from the questions raised previously, you have to run some experiments (just a few ones)
    to check if some improvements can be obtained. Report it in the document you have to prepare for this
    lab practice, and if such findings are relevants, include them in the presentation.

    Report and presentation for this lab practice must focus on the effect of the above mentioned
    configuration hyperparameters in terms of performance and running time (computational cost)
    for the two datasets used in this lab practice.
