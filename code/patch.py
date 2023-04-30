from sklearn.preprocessing import QuantileTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def partition_data(group_num, df):
    partitioned = [df.iloc[i:i+group_num] for i in range(0, len(df), group_num)]
    return partitioned

def collect_info_partition(df, y, index):
    col_names = df[index].columns
    features = [i for i in col_names if i != y]
    return {col: df[index][col].value_counts().to_dict() for col in features}

def collect_info_all(dict_collect):
    combined_dict = defaultdict(lambda: defaultdict(int))
    for dictionary in dict_collect:
        for key, value in dictionary.items():
            for sub_key, sub_value in value.items():
                # Add the sub_value to the corresponding key in the combined_dict
                combined_dict[key][sub_key] += sub_value
    result_dict = dict(combined_dict)
    return {key: dict(map(lambda x: (x[0], x[1]), value.items())) for key, value in result_dict.items()}

def bin_method_syn(feature_dict, num_bins):
    values = sorted(list(feature_dict.keys()))
    bin_edges = np.linspace(start=min(values), stop=max(values), num=num_bins+1 if len(set(values)) > num_bins else (len(set(values)) + 1))
    hist, _ = np.histogram(values, bins=bin_edges)
    bdd =  _[1:-1]
    return bdd

def ma_syn(feature_dict, window):
    value = sorted(list(feature_dict.keys()))
    return np.convolve(value, np.ones(window), 'valid') / window

def get_datatype(df):
    cate_type = []
    for feature in df.columns:
        if df[feature].dtype == 'O':
            cate_type.append(feature)
    return cate_type

def traverse_point(dict_all_collect, threshold, num_bins, window, method, cate_type, feature):
    traverse = {}
    if len(dict_all_collect[feature]) < threshold:
        traverse[feature] = list(dict_all_collect[feature].keys())
    else:
        # method: bin, ma(moving avg)
        if feature not in cate_type:
            if method == 'bin':
                traverse[feature] = bin_method_syn(dict_all_collect[feature], num_bins)
            else:
                traverse[feature] = ma_syn(dict_all_collect[feature], window)
        else:
            traverse[feature] = list(dict_all_collect[feature].keys())
    return traverse

def get_feature(df, y):
    return [i for i in df.columns if i != y]

def cat_to_num(labels, df, target):
    binary_codes = {labels[0]: 0, labels[1]: 1}
    y = df[target]
    encoded_values = [binary_codes[val] for val in y]
    return encoded_values

def basic_eda(df):
    for col in df.columns:
        fig, axs = plt.subplots(ncols=2, figsize=(10,5))
        # plt.margins(x=5)
        if not df[col].dtype == 'O':
            sns.histplot(df, x = col, hue = 'income', ax = axs[0])
            axs[0].set_xticklabels(axs[0].get_xticks(),rotation=45)
            sns.boxplot(df, x = col, y = 'income', ax = axs[1])
            axs[1].set_title(f'Boxplot of {col}')
        else:
            sns.countplot(df, x = col, hue = 'income', ax = axs[0])
            axs[0].set_xticklabels(axs[0].get_xticks(),rotation=45)
        axs[0].set_title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.show()
        
def qtr(df, n, dist):
    qt = QuantileTransformer(n_quantiles=n, output_distribution=dist)
    for col in df.columns:
        if not df[col].dtype == 'O':
            df[col] = qt.fit_transform(np.array(df[col]).reshape(-1,1))