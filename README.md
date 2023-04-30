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

###Goal: 
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
  DTC implementation code
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
  - Bin number method specified by user, with values can be `sturges, scott, tanh, logistic`. Read more specific meaning in [report](https://github.com/yutingmeivu/DTC-MapReduce/blob/main/DTCMR.pdf).
  - ###### `rule` (string)
  - Not specified by user
  
Code and dataset is also avaliable in Google Colab:
### [Open in Colab](https://drive.google.com/drive/folders/1BU97Eyspj8umJahqHWMd2Nyq5MFobw9K?usp=share_link)
