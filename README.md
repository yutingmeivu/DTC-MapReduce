# Decision Tree Classification Model with Map Reduce

This is a project for designing and implementing Decision Tree Classification(DTC) model
by building from scratch and use map reduce strategy for a prediction task on ‘Adult’dataset.
Possible scenario for model implementation is assumed and model performance under certain
setting is measure

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Examples](#examples)

## Introduction
This is a project for designing and implementing Decision Tree Classification(DTC) model
by building from scratch and use map reduce strategy for a prediction task on ‘Adult’dataset.
Possible scenario for model implementation is assumed and model performance under certain
setting is measured.

## DTC structure

### Goal: 
The overall goal of DTC is to find best splits at each node which get informative and effective
tree in unseen dataset, with both maximizing infomation gain to control overall reduction
in uncertainty and minimizing impurity of each node with accurate label for controlling
homogeneity of labels in subsets. We will assume DTCMR is using under big data set in limited memory usage.

### Stopping criteria:
1. Tree Depth < maximum depth
2. Samples in Node ≥ minimum number of sample

### Impurity Measure:
- Gini impurity: $1 - \sum_{i = 1}^k p_i^2$, $k$ is the number of labels in target variable.
- Variance: $\boldsymbol{E}[Y^2] - \boldsymbol{E}[Y]^2.$

## DTCMR

The MapReduce strategy divides a complex task into smaller sub-tasks that can be processed
independently in parallel across a cluster of computers. The process consists of two main
stages: the map stage and the reduce stage. We designed pipeline with using map reduce under DTC model as follows:
![pipeline for DTCMR](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/map.png)


## Usage

### [tree.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/tree.py)
  DTC implementation
  #### Parameters introduction in Node
  - ###### `X` (DataFrame)
  - Dataset for training excluding target variable
  - ###### `Y` (list)
  - Target variable in training dataset
  - ###### `labels` (list)
  - Label categories in Y
  - ###### `outlier_` (Bool)
  - True if removing outliers
  - ###### `traverse_threshold` (int)
  - Threshold for number of partition points. For discrete variable, if set of distinct values exceed, set it as maximum number of partition points. For continuous variable, bound the minimum partition number.
  - ###### `min_sample_split` (int)
  - Threshold for minimum samples in each node with splitting, as one of stopping criteria
  - ###### `max_depth` (int)
  - Threshold for maximum number of depth specified by user, as one of stopping criteria
  - ###### `node_type` (string)
  - Not specified by user
  - ###### `depth` (int)
  - Not specified by user
  - ###### `na_threshold` (float)
  - range from [0,1]. Threshold for determine if drop feature when percentage of missing values exceeds threshold.
  - ###### `info_method` (string)
  - `variance, gini`. Impurity function specified by user.
  - ###### `na_method` (string)
  - `mean, median, recursive`. Missing value imputation method specified by user, default as `mean`. Notice that if feature is normally distributed, result from three methods would be similar.
  - ###### `bins` (int, string)
  - Bin number method specified by user, with values can be either number or `sturges, scott, tanh, logistic`. Read more specific meaning in [report](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/DTCMR.pdf).
  - ###### `rule` (string)
  - Not specified by user

### [treemr.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/treemr.py)
  DTCMR implementation
  #### Parameters introduction in Node_parallel
  Params with same name as Node has the same meaning, so just skiped duplicated ones.
  - ###### `features_df, cate_type, traverse_all` (list)
  - Global variable, not specified by user. `features_df`: features for traverse. `cate_type`: list of categorical variable names. `traverse_all`: traverse point with respective of features grasped from information in original overall dataset using map reduce strategy before implementing tree growth.
  - ###### `method` (string)
  - Fixed as `bin` method in DTCMR due to consideration of complexity and latent inconsistent information from subgroup if implementing other bin number generation method.
  - ###### `window` (int)
  - Time window for moving average for numerical variable.
  - ###### `threshold` (int)
  - same meaning as `traverse_threshold` in [tree.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/tree.py), used for threshold in subdata groups.
  
### [run.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/run.py)
  DTC model main function for running using `run_raw`
  #### Parameters:
  Skip the params introduced in [tree.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/tree.py)
  - `fill` (list)
  - Fill NA in test dataset based on output in running `run_raw`. Result of missing value imputation in training set is printed out after instantiating an object in Node and use `.grow_tree()`.
  #### Expected output:
  - Print out Max precision score, Mean precision score with standard deviation, Mean computational time(seconds) per fold.
  
### [run_mr.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/run_mr.py)
  DTCMR main function for running using `run_code`
  #### Parameters:
  - `d_s` (DataFrame)
  - Dataset including all features and target variable without partition. (🥲 I forgot to set target variable name as a param, anyway it's not that important..)
  - `sub_index`
  - Less than or equal to number of observations in overall dataset, if want to test model performance in subset.
  - `group` (int)
  - Number of subsamples inside a group. 
  - `method` (string)
  - Impurity function name `variance, gini`, as same as `info_method` in [tree.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/tree.py).
  
### [patch.py](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/code/patch.py)
  A hodgepodge collection of functions for either for getting statistics from overall dataset using map reduce before tree start growing or EDA and preprocessing of original dataset.
  
### Examples
To successfully running code in local, be sure to import following packages:
```python
import os
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from patch import partition_data, collect_info_partition, collect_info_all, bin_method_syn,\
ma_syn, get_datatype, traverse_point, get_feature, cat_to_num, basic_eda, qtr
from run import run_raw
from run_mr import run_code, get_distinct, get_preprocess, label_, pre_group
```
Use the following code for running DTC under certain setting for dataset `dt` under repeated stratified k fold cross validation, notice that DTC accept both numerical and categorical variables:
```python
run_raw(dt, n_splits = 10, n_repeats = 1, traverse_threshold = 25, min_samples_split = 1850, \
        max_depth = 11, info_method = 'variance', na_method = 'recursive', bins = 'tanh', outlier_=False,\
       fill = ['Self-emp-not-inc', 'Other-service', 'South'])
```
Use the following code format for running DTCMR under train test split with `d_s` dataset:
```python
pool = mp.Pool(20)
precisionl = []
endl = []
subsample_num = 1000
group = 500
method = 'variance'
for i in range(10):
    precision, end = run_code(d_s, subsample_num, group, method)
    precisionl.append(precision)
    endl.append(end)
mpr = np.mean(precisionl)
mt = np.mean(endl)
pool.close()
print(f'mean computation time {round(mt, 3)} seconds with mean precision at {round(mpr, 3)} with total {subsample_num} samples with groups S = {group} with impurity function as {method}.')
```
  
## Google Colab
Code and dataset is also avaliable in [Google Colab](https://drive.google.com/drive/folders/1BU97Eyspj8umJahqHWMd2Nyq5MFobw9K?usp=share_link)! 🤩 Try [DTCMR.ipynb](https://colab.research.google.com/drive/11ADyP9g0bR7YR1RxdI_8F1mhYx9PxGqN)!
