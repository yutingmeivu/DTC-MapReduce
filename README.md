# Comparision of performance of nonlinear optimization method used in accelerating multi neuron network

we compared two optimization method: conjugate gradient method with classic gradient descent in fully connected multi neuron network. Newton's method, regula falsi and secant method would be covered. 

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)

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



## Usage

Code and dataset is avaliable in Google Colab:
### [Open in Colab](https://drive.google.com/drive/folders/1BU97Eyspj8umJahqHWMd2Nyq5MFobw9K?usp=share_link)
