# Lab practice 3

Lab practice 3 is about two use cases:

1. [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

2. Use Case 13 of the [DeepHealth EU project](https://deephealth-project.eu)
   based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0),
   and already presented in [Lab practice 1](../pract_1)


## Work to do in this lab practice

1. Accessing to the cluster from a computer in the UPVNET domain or via Virtual Private Network (VPN) previously configured

    >
    >    `ssh -YC jon@teaa-master-cluster.dsicv.upv.es`
    >

    Use your username instead of `jon`

2. Interaction with the Hadoop Distributed File System (HDFS)

    Once in the system, the first step is to see what we can find in the HDFS.
    The datasets to be used are in the directory of the `ubuntu` user, so we have
    to use the absolute path to see what it contains: `/user/ubuntu`.

    >
    > `hdfs dfs -ls -h /user/ubuntu/data`
    >
    > `hdfs dfs -ls -h /user/ubuntu/data/uc13`
    >

3. Creation of local dirs in user home directory

   In this lab practice, you do not need to worry about this step,
   the code will create the directories automatically
   starting from your home directory.

   The three directories that will created are `digits`, `uc13.1` and `uc13-21x20`

4. Creation of the user directory in the HDFS

   For this lab practice, it is not necessary to create new directories in the HDFS.

5. Two sets of experiments must be carried out in this lab practice, one with KDE and another with KNN.

    For each set of experiments, there are three subsets, one based on the
    [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database),
    and two based on the [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

    1. KDE based on [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

        Set of experiments where `bandwidth` and the PCA number of components are varied.
        [Here](experiments/kde-mnist.csv) you can find the experiments to
        be executed assigned to each group. Your group number can be found in the
        document available in **PoliformaT**.

    2. KDE based on [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

        Set of experiments where `bandwidth` and the codebook size to do a KMeans per class are varied
        (codebook size equal to zero means to do not use KMeans).
        Additionally, label reduction is also applied, one half of the experiments is done with label reduction,
        and the other half without it. 
        You can see in the [CSV file](experiments/kde-uc13.csv) that the same combination of hyper-parameters is used for different patients.

        In this set of experiments per patient the original data preprocessing used in lab practice 1 is used after applying PCA to reduce
        from 249 (21x14) dimensions to 29.

    3. KDE based on [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

        Analogously, similar experiments are done in the case of a new data preprocessing step, where the 21 channels are
        kept independently, but the classification of each sample is done by combining the classification of the 21 20-dimensional
        vectors of each channel. In this case, the 20-dimensional vector of each channel is the output of the filter bank,
        so all are frequency domain variables covering from less than 1Hz to around 128 Hz, but fine-grained 
        at low frequencies and coarse-grained at high frequencies.
        
        In this case no KMeans codebook size can be especified,
        it has been force to be always 4000,
        and also label reduction is mandatory, otherwise the running time is prohibitive.
        So, only the `bandwidth` and the patient can be varied, you can find the set of experiments per group [here](experiments/kde-uc13-21x20.csv)


    4. KNN based on [MNIST Digits database](https://en.wikipedia.org/wiki/MNIST_database)

        Set of experiments analogous to the KDE with MNIST but varying `K`, the number of nearest neighbours, instead of the `bandwidth`.
        [Here](experiments/knn-mnist.csv) you can find the list of experiments and the assigment to each group.

    5. KNN based on [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

        Set of experiments similar to the one done previously with this dataset using KDE with PCA to reduce dimensionality to 29,
        but in this case no KMeans is used and `K`, the number of nearest neighbours,
        must be varied, instead of the `bandwidth`.
        [Here](experiments/knn-uc13.csv) you can find the list of experiments and the assigment to each group.

    6. KNN based on [CHB-MIT Scalp EEG Database](https://physionet.org/lightwave/?db=chbmit/1.0.0)

        Set of experiments similar to the one done previously with this dataset using KDE with no PCA and working with each channel independently,
        but in this case no KMeans is used and `K`, the number of nearest neighbours,
        must be varied, instead of the `bandwidth`.
        [Here](experiments/knn-uc13-21x20.csv) you can find the list of experiments and the assigment to each group.


6. What the execution of the different sets of experiments has created

    The models, it it is the case, have been stored in the local file system, as the results.
    You can explore what has been created in your home directory using the following commands,
    or refinements from them:

    >
    > `tree ${HOME}/digits`
    >
    > `tree ${HOME}/uc13.1`
    >
    > `tree ${HOME}/uc13-21x20`
    >

7. The results can be seen from the text file complementary to the figures

    Some examples:

    > 
    > `cat ${HOME}/digits/results.3.train/kde-classification-results-pca-30-bw-1.000.txt`
    >

    > 
    > `cat ${HOME}/uc13.1/chb03/results.4-train/kde-classification-results-bw-2.000.txt`
    >

    > 
    > `cat ${HOME}/uc13-21x20/chb03/results.4-train/kde-classification-results-bw-2.000.txt`
    >

    You can see the images with `viewnior`, for example:

    >
    > `viewnior ${HOME}/digits/results.3.train/kde-classification-results-pca-30-bw-1.000.png`
    >
    > `viewnior ${HOME}/digits/results.3.train/kde-classification-results-pca-30-bw-1.000.svg`
    >

    > 
    > `viewnior ${HOME}/uc13.1/chb03/results.4-train/kde-classification-results-bw-2.000.png`
    >
    > `viewnior ${HOME}/uc13.1/chb03/results.4-train/kde-classification-results-bw-2.000.svg`
    >

    > 
    > `viewnior ${HOME}/uc13-21x20/chb03/results.4-train/kde-classification-results-bw-2.000.png`
    >
    > `viewnior ${HOME}/uc13-21x20/chb03/results.4-train/kde-classification-results-bw-2.000.svg`
    >

8. Report and presentation preparation

    As you can guess from the CSV files with lists of experiments assigned to different groups,
    once you all completed all the experimentation, all the results should be shared between
    all the groups, so that each group will has to analize all the results in order to
    study the effect of several configuration hyper-parameters, as the `bandwidth`, `K`, the
    number of k-nearest neighbours, and when it is the case, the `codebook size` to apply KMeans
    in order to reduce the size of the training set.

    Report and presentation for this lab practice must focus on the effect of the above mentioned
    configuration hyperparameters for the two datasets used in this lab practice.
