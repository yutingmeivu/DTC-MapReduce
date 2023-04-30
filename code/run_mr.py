from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
from treemr import Node_parallel
import multiprocessing as mp
from collections import defaultdict
from collections import Counter
import math
import numpy as np
from patch import partition_data, collect_info_partition, collect_info_all, bin_method_syn,\
ma_syn, get_datatype, traverse_point, get_feature, cat_to_num, basic_eda, qtr

global cate_type
global traverse_all
global features_df

def get_distinct(d):
    distinct_values = {}
    for column in d.columns:
        if (d[column].dtype == 'O'):
            distinct_values[column] = d[column].unique().tolist()
        else:
            distinct_values[column] = [(min(d[column]), max(d[column]))]
    return distinct_values

def get_preprocess(d_s):
    cate_type = get_datatype(d_s)
    features_df = get_feature(d_s, 'income')
    distinct_values = get_distinct(d_s)
    cols = cate_type
    for i in cols:
        label_(i, d_s)
    return features_df, cate_type

def label_(col, df):
    distinct_values = get_distinct(df)
    rel = distinct_values[col]
    rel = [i for i in rel if i is not None]
    rel_level = {}
    for i in range(len(rel)):
        rel_level[rel[i]] = i + 1

    df[col] = df[col].map(rel_level)
    
def pre_group(group_num, d_s):
    features_df, cate_type = get_preprocess(d_s)
    dict_d = []
    target = 'income'
    y = d_s[target]
    labels = sorted(list(set(y)))
    d_s[target] = cat_to_num(labels, d_s, target)

    pool = mp.Pool(20)
    d_p = partition_data(group_num, d_s)
    dict_d = pool.starmap_async(collect_info_partition, [(d_p, 'income', i) for i in range(len(d_p))]).get()

    dict_f = collect_info_all(dict_d)

    pool = mp.Pool(20)
    traverse_all = pool.starmap_async(traverse_point, [(dict_f, 12, 20, None, 'bin', cate_type, feature) for feature in features_df]).get()
    traverse_all = {k: v for d in traverse_all for k, v in d.items()}
    pool.close()
    return features_df, cate_type, traverse_all

def run_code(d_s, sub_index, group_num, method):
    features_df, cate_type, traverse_all = pre_group(group_num, d_s)
    train_df, test_df = train_test_split(d_s.loc[:sub_index, :], test_size=0.2,
                                         stratify=d_s.loc[:sub_index, :]['income'])
    d_ptr = partition_data(group_num, train_df) 
    # d_ptr = partition_data
    target = "income"
    y = d_s[target]
    labels = sorted(list(set(y)))
    test_df_ = test_df.copy()
    test_df_.drop(target, axis=1)
    y_test = test_df[target]
    custom_tree_start = time.time()
  
    root = Node_parallel(d_ptr, target, labels, features_df, cate_type, traverse_all,
                         min_samples_split=600,max_depth=11, node_type=None, depth=None,
                         info_method=method,rule=None, method = 'bin', num_bins = 15, 
                         window = None, threshold = 10)
    root.grow_tree()
    # root.print_tree()


    # Evaluate the precision of the model on the test data
    y_pred = root.predict(test_df_)

    precision = precision_score(list(test_df['income']), y_pred)

    end = time.time() - custom_tree_start
    # mean_precision = np.mean(precision_scores)
    # std_precision = np.std(precision_scores)
    # max_precision = np.max(precision_scores)

    # 8 mins
    print(f"precision score: {precision:.3f} with {end} seconds")
    return precision, end