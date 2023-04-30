from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
from tree import Node

def run_raw(dt, n_splits, n_repeats, traverse_threshold, min_samples_split, max_depth, info_method, na_method, bins, outlier_, fill):
    # Define the number of folds and repeats
    # n_splits = 10
    # n_repeats = 2

    # Create the cross-validation iterator
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    # Define the features and target variable
    target = "income"
    cols = dt.columns
    features = [i for i in cols if i != target]
    X = dt[features]
    y = dt[target]

    Y = dt[target].values.tolist()
    labels = sorted(list(set(Y)))

    # Initialize a list to store the precision scores
    precision_scores = []


    custom_tree_start = time.time()

    for train_idx, test_idx in cv.split(X, y):
        # Split the data into training and test sets
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Fit the decision tree on the training data
        root = Node(X_train, y_train, labels, outlier_=outlier_, traverse_threshold=traverse_threshold, min_samples_split=min_samples_split,
                max_depth=max_depth, node_type=None, depth=None, na_threshold=0.9, info_method=info_method,
                na_method=na_method, bins=bins, rule=None)
        root.grow_tree()

        X_test['workclass'] = X_test['workclass'].fillna(fill[0])
        X_test['occupation'] = X_test['occupation'].fillna(fill[1])
        X_test['native.country'] = X_test['native.country'].fillna(fill[2])

        # Evaluate the precision of the model on the test data
        y_pred = root.predict(X_test)
        label_map = {'<=50K': 0, '>50K': 1}
        y_test_trans = y_test.map(label_map)
        y_pred_trans = pd.Series(y_pred).map(label_map)
        precision = precision_score(y_test_trans, y_pred_trans)
        precision_scores.append(precision)

    end = (time.time() - custom_tree_start) / (n_splits * n_repeats)
    mean_precision = np.mean(precision_scores)
    std_precision = np.std(precision_scores)
    max_precision = np.max(precision_scores)

    print(f"Max precision score: {max_precision:.3f}, Mean precision score: {mean_precision:.3f} (std={std_precision:.3f}) with {end} seconds per fold")
