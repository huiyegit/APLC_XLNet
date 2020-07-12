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

1. Download our preprocessed data for  and save them to `data/`.
2. Unzip the zip files.
