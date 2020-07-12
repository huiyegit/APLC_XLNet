## APLC_XLNet

This is the official Pytorch implementation of the paper [Pretrained Generalized Autoregressive Model with Adaptive Probabilistic Label Clusters for Extreme Multi-label Text Classification](http://arxiv.org/abs/2007.02439)


The source code will be uploaded soon!

## Requirements
* Linux
* Python ≥ 3.6
    ```bash
    # We recommend you to use Anaconda to create a conda environment 
    conda create --name aplc_xlnet python=3.6
    conda activate aplc_xlnet
    ```
* PyTorch ≥ 1.4.0
    ```bash
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
* Other requirements:
    ```bash
    pip install -r requirements.txt
    ```
## Prepare Data

### preprocessed data

1. Download our preprocessed data for [ERUlex](https) [Wiki10](https:) [AmazonCat](https:) [Wiki500k](https:) [Amazon670k](https:) and save them to `data/`.
2. Unzip the zip files.

### raw texts
Our raw texts are from [ERUlex](https:) [Wiki10](https:) [AmazonCat](https:) [Wiki500k](https:) [Amazon670k](https:)

### preprocess your custom dataset
1. Create the train.csv file for training and dev.csv file for testing. Reference our preprocessed dataset for the format of the csv file
2. Create the labels.txt file. Labels should be sorted in descending order accoridng to their frequency
3. Count the largest number of positive labels of one sample in all samples, and assign this value to the hyperparameter --pos_label
4. Add your dataset name into the dictionary 'processors' and 'output_modes' in the source file 'the utils_multi_label.py'
5. Create your bash file and set the hyperparameters in 'code/run/'

## Training and evaluation
Run the command for the dataset EURlex
```
bash ./run/eurlex.bash
```
## Training and evaluation
Run the command for the dataset EURlex
```
bash ./run/eurlex.bash
```
## Evaluation on our trained models
1. Download our trained models for 

2. Run the command for the dataset EURlex
```
bash ./run/eurlex.bash
```
