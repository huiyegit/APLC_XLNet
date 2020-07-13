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

Download our preprocessed datasets from [Google Drive](https://drive.google.com/drive/folders/1bRLrc8N3ukzAVn9zyTqr0IqP3fWJUYAt?usp=sharing) and save them to `data/`

### preprocess the custom dataset
1. Create `train.csv` and `dev.csv`. Reference our preprocessed dataset for the format of the csv file
2. Create `labels.txt`. Labels should be sorted in descending order accoridng to their frequency
3. Count the number of positive labels of each sample, select the largest number in all samples, and assign this value to the hyperparameter `--pos_label`
4. Add the dataset name into the dictionary `processors` and `output_modes` in the source file `utils_multi_label.py`
5. Create the bash file and set the hyperparameters in `code/run/`

## Run the source code
### Training and evaluation
Run the commands
- For dataset EURlex:     `bash ./run/eurlex.bash`
- For dataset Wiki10:     `bash ./run/wiki10.bash`
- For dataset AmazonCat:  `bash ./run/amazoncat.bash`
- For dataset Wiki500k:   `bash ./run/wiki500k.bash`
- For dataset Amazon670k: `bash ./run/amazon670k.bash`

### Evaluation on our pretrained models
1. Download our pretrained models from [Google Drive](https://drive.google.com/drive/folders/1SK2OO6tqxxYZBCkQOVsULzdEy_ZyKUd8?usp=sharing)  and save them to `models/`

2. Run the commands 
   - For dataset EURlex:     `bash ./run/eurlex_evaluate.bash`
   - For dataset Wiki10:     `bash ./run/wiki10_evaluate.bash`
   - For dataset AmazonCat:  `bash ./run/amazoncat_evaluate.bash`
   - For dataset Wiki500k:   `bash ./run/wiki500k_evaluate.bash`
   - For dataset Amazon670k: `bash ./run/amazon670k_evaluate.bash`
