from collections import Counter
import math
import numpy as np
import math
import pandas as pd
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Node:
    def __init__(self, X: pd.DataFrame, Y: list, labels: list, outlier_ = None, traverse_threshold = None, min_samples_split=None, max_depth=None, node_type=None, \
               depth = None, na_threshold = 0.9, info_method = 'variance', na_method = 'mean', bins = None, rule=None):
        self.depth = depth if depth else 0
        self.X = X
        self.Y = Y
        self.features = list(self.X.columns)
        self.counts = Counter(Y)
        self.labels = labels
        self.na_threshold = na_threshold
        self.info_method = info_method
        self.na_method = na_method
        self.bins = bins if bins else 'sturges'
        self.n = len(Y)
        self.gini_impur = self.get_gini()
        self.variance = self.get_var()
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 15
        self.node_type = node_type if node_type else 'root'
        self.rule = rule if rule else ""
        self.traverse_threshold = traverse_threshold if traverse_threshold else 5
        self.outlier_ = outlier_ if outlier_ else False

        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        self.yhat = yhat 

        self.left = None 
        self.right = None 
        self.best_feature = None 
        self.best_value = None 
        
    @staticmethod
    def prob_compute(y1_count, y2_count):
        y1_count = 0 if y1_count is None else y1_count
        y2_count = 0 if y2_count is None else y2_count
        n = y1_count + y2_count
        p1 = y1_count / n if n != 0 else 0
        p2 = y2_count / n if n != 0 else 0
        return p1, p2

    def cat_to_num(labels, df):
        binary_codes = {labels[0]: 0, labels[1]: 1}
        y = df['Y']
        encoded_values = [binary_codes[val] for val in y]
        return encoded_values

    def gini_impurity(self, y1_count, y2_count):
        # y1_count, y2_count: the # of corresponding label of y
        p1, p2 = self.prob_compute(y1_count, y2_count)
        gini = 1 - p1**2 - p2**2
        return gini
  
    @staticmethod
    def vari(x, labels):
        dict_label = {labels[0]: 0, labels[1]: 1}
        x_trans = [dict_label[item] for item in x]
        n = 0 if len(x_trans) is None else len(x_trans)
        q = list(map(lambda x: x**2, x_trans))
        s = sum(x_trans)
        var = 1/n * (sum(q)) - (s / n)**2 if n != 0 else 0
        return n * var

    def get_gini(self):
        y1_count, y2_count = self.counts[self.labels[0]], self.counts[self.labels[1]]
        return self.gini_impurity(y1_count, y2_count)

    def get_var(self):
        vari_ = self.vari(self.Y, self.labels)
        return vari_

    def set_point(self, df_nna, feature, best_value):
        if type(best_value) == str:
            left_counts = Counter(df_nna[df_nna[feature] == best_value]['Y'])
            right_counts = Counter(df_nna[df_nna[feature] != best_value]['Y'])
        else:
            left_counts = Counter(df_nna[df_nna[feature] >= best_value]['Y'])
            right_counts = Counter(df_nna[df_nna[feature] < best_value]['Y'])
        p_left_label0, p_left_label1 = self.prob_compute(left_counts[self.labels[0]], left_counts[self.labels[1]])
        p_right_label0, p_right_label1 = self.prob_compute(right_counts[self.labels[0]], right_counts[self.labels[1]])
        compare_prob = [p_left_label0, p_left_label1, p_right_label0, p_right_label1]
        max_index = compare_prob.index(max(compare_prob))
        return max_index, max(compare_prob)
  
    def missing_value(self, df, feature):
        percentage = df[feature].isna().sum() / len(df)
        if isinstance(percentage, pd.Series):
            percentage = list(percentage)[0]

        # percentage = df[df[feature] == '?'][feature].count() / len(df)
        df['Y'] = self.Y
        df_nna = df[df[feature].notnull()][[feature, 'Y']]
        df_nna = df_nna.sort_values(by=feature)

        best_feature, best_value, traverse, df_nna = self.best_split(df_nna, [feature], None, True)
        last_value = best_value

        if (self.na_method != 'recursive') & ((df_nna[feature].dtype != 'O') | (df_nna[feature].dtype != 'string')):
            max_index, max_prob = self.set_point(df_nna, feature, best_value)
            if self.na_method == 'mean':
                if max_index <= 1:
                    fill_na = df_nna[df_nna[feature] >= best_value][feature].mean()
                else:
                    fill_na = df_nna[df_nna[feature] < best_value][feature].mean()
            if self.na_method == 'median':
                if max_index <= 1:
                    fill_na = df_nna[df_nna[feature] >= best_value][feature].median()
                else:
                    fill_na = df_nna[df_nna[feature] < best_value][feature].median()
            else:
                if (df_nna[feature].dtype == 'O') | (df_nna[feature].dtype == 'string'):
                    value_cum = []
                    max_index, max_prob = self.set_point(df_nna, feature, best_value)
                    if max_index <= 1:
                        # fill_na = df_nna[df_nna[feature] == best_value][feature][0]
                        fill_na = best_value
                    else:
                        # value_cum.append(best_value)
                        # traverse = set(traverse) - set(value_cum)
                        traverse = [i for i in traverse if i != best_value]
                        while len(traverse) > 1:
                            # parent = self.vari(df_nna['Y'], self.labels)
                            best_feature, best_value, traverse, df_ = self.best_split(df_nna, [feature], traverse, True)
                            max_index, max_prob = self.set_point(df_nna, feature, best_value)
                            if max_index <= 1:
                                break
                            else:
                                # value_cum.append(best_value)
                                # traverse = set(traverse) - set(value_cum)
                                # traverse = list(traverse)
                                tmp = [i for i in traverse if i != best_value]
                                if len(tmp) > 1:
                                    traverse = tmp 
                                else:
                                    traverse = [best_value]
                        fill_na = best_value
                
        else:
            value_cum = []
            max_index, max_prob = self.set_point(df_nna, feature, best_value)
            last_prob = max_prob
            # last_value = best_value
            traverse = [i for i in traverse if i > best_value]
            if max_prob <= 1:
                if len(traverse) == 0:
                    traverse = [best_value]
                else:
                    # df_nna = df_nna[df_nna[feature] > best_value]
                    if df_nna is None:
                        traverse = [best_value]
            else:
                tmp = [i for i in traverse if i < best_value]
                if len(tmp) > 1:
                    traverse = tmp 
                else:
                    traverse = [best_value]
                    # df_nna = df_nna[df_nna[feature] < best_value]
            while len(traverse) > 1:
                # parent = self.vari(df_nna['Y'], self.labels)
                best_feature, best_value, traverse, df_ = self.best_split(df_nna, [feature], traverse, True)
                max_index, max_prob = self.set_point(df_nna, feature, best_value)
                if max_index <= 1:
                    if last_prob >= max_prob:
                        traverse = [last_value]
                        best_value = last_value
                        break
                    else:
                        traverse = [i for i in traverse if i > best_value]
                        # df_nna = df_nna[df_nna[feature] >= best_value]
                else:
                    tmp = [i for i in traverse if i < best_value]
                    if len(tmp) > 1:
                        traverse = tmp 
                    else:
                        traverse = [best_value]
                        break
                    # df_nna = df_nna[df_nna[feature] < best_value]
                    last_prob = max_prob
                    last_value = best_value
                    # print(best_feature, best_value, len(traverse), traverse)
            fill_na = best_value
        return fill_na                

    @staticmethod
    def ma(x: np.array, window: int):
        return np.convolve(x, np.ones(window), 'valid') / window
  
    @staticmethod
    def bin_method(x, num_bins):
        bin_edges = np.linspace(start=min(x), stop=max(x), num=num_bins+1 if len(set(x)) > num_bins else (len(set(x)) + 1))
        hist, _ = np.histogram(x, bins=bin_edges)
        bdd =  _[1:-1]
        return bdd

    @staticmethod
    def outlier(x):
        # histogramming in this way would be sensitive to outliers
        # more suitable on the case when there's no imbalanced class in feature
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        outliers = [i for i in x if ((i < lower_bound) or (i > upper_bound))]
        rm_out = [i for i in x if i not in outliers]
        return rm_out
  
    def info_gain(self, parent, left_counts, right_counts, left_y, right_y):
        y0_left, y1_left, y0_right, y1_right = left_counts.get(self.labels[0],0), left_counts.get(self.labels[1],0),\
         right_counts.get(self.labels[0],0), right_counts.get(self.labels[1],0)
        if self.info_method == 'gini':
            gini_left = self.gini_impurity(y0_left, y1_left)
            gini_right = self.gini_impurity(y0_right, y1_right)

            n_left = y0_left + y1_left
            n_right = y0_right + y1_right

            w_left = n_left / (n_left + n_right)
            w_right = n_right / (n_left + n_right)

            wGINI = w_left * gini_left + w_right * gini_right

            gain = parent - wGINI
        else:
            vari_left = self.vari(left_y, self.labels)
            vari_right = self.vari(right_y, self.labels)

        gain = parent - (vari_left + vari_right)
      
        return gain

    @staticmethod
    def trace_gain(max_gain, gain_temp, feature, value, info_method, best_feature, best_value):
        # if info_method == 'gini':
        if gain_temp > max_gain:
            best_feature = feature
            best_value = value 
            max_gain = gain_temp
        return max_gain, best_feature, best_value

    @staticmethod
    def logistic_(x):
        return 2*((1 / (1 + np.exp(-x/2))) - 0.5)
  
    @staticmethod
    def tanh_(x):
        return math.tanh(x)

    def best_split(self, df = None, features = None, traverse = None, outlier_ = None):
        # measure: 'gini', 'variance'
        df = self.X.copy() if df is None else df
        # features_copy = df.columns
        if 'Y' not in df.columns:
            df['Y'] = self.Y
        labels = self.labels
        info_method = self.info_method
        parent = self.get_gini() if info_method == 'gini' else self.get_var()
        # parent = self.gini_impurity if info_method == 'gini' else self.variance

        max_gain = None

        best_feature = None
        best_value = None

        traverse_threshold = self.traverse_threshold
    
        features = self.features if features is None else features
        traverse = traverse if traverse else None
        rm_outlier = outlier_ if outlier_ else self.outlier_

        Xdf = df.copy()

        for feature in features:
            fill_na = None
            percentage = Xdf[feature].isna().sum() / len(Xdf)
            if isinstance(percentage, pd.Series):
                percentage = list(percentage)[0]
            # percentage = Xdf[Xdf[feature] == '?'][feature].count() / len(Xdf)
            if percentage > self.na_threshold:
                continue
            else:
                if percentage != 0:
                    fill_na = self.missing_value(Xdf, feature)
                    Xdf[feature] = Xdf[feature].fillna(fill_na)
                    print("="*10)
                    print(f"missing value of feature {feature} with imputation of {fill_na}.")
                    print("="*10)
                if Xdf is not None:
                    if (Xdf[feature].dtype == 'O') | (Xdf[feature].dtype == 'str'):
                        # categorical
                        if traverse is None:
                            traverse = Xdf[feature].unique()
                            traverse = sorted(traverse)
              
                        for value in traverse:
                            left_counts = Counter(Xdf[Xdf[feature] == value]['Y'])
                            right_counts = Counter(Xdf[Xdf[feature] != value]['Y'])
                            left_y = Xdf[Xdf[feature] == value]['Y']
                            right_y = Xdf[Xdf[feature] != value]['Y']
                
                            gain_temp = self.info_gain(parent, left_counts, right_counts, left_y, right_y)
                            if best_feature is None:
                                best_feature = feature
                            if best_value is None:
                                best_value = value
                            if max_gain is None:
                                max_gain = gain_temp
                
                            max_gain, best_feature, best_value = self.trace_gain(max_gain, gain_temp, feature, value, self.info_method, best_feature, best_value)  
                  

                    else:
                        # discrete or continuous
                        if traverse is None:
                            tmp = len(Xdf[feature].unique())
                        else:
                            tmp = len(traverse)
                        if tmp <= traverse_threshold:
                            traverse = self.ma(sorted(Xdf[feature].unique()), 2) if traverse is None else self.ma(sorted(traverse), 2) 
                        else:
                            if self.bins == 'sturges':
                                num_bins = int(np.ceil(np.sqrt(len(Xdf[feature])))) 
                            elif self.bins == 'scott':
                                bin_width = 3.5 * np.std(Xdf[feature]) / np.power(len(Xdf[feature]), 1/3)
                                num_bins = int(np.ceil((np.max(Xdf[feature]) - np.min(Xdf[feature])) / bin_width))
                            elif self.bins == 'logistic':
                                tmp_length = self.ma(sorted(Xdf[feature].unique()), 2) if traverse is None else self.ma(sorted(traverse), 2) 
                                num_bins = int(np.ceil(self.logistic_(parent) * np.sqrt(len(tmp_length))))
                                num_bins = max(num_bins, self.traverse_threshold) 
                                # default = len(self.ma(sorted(Xdf[feature].unique()), 2)) if traverse is None else len(self.ma(sorted(traverse), 2))
                            elif self.bins == 'tanh':
                                tmp_length = self.ma(sorted(Xdf[feature].unique()), 2) if traverse is None else self.ma(sorted(traverse), 2) 
                                num_bins = int(np.ceil(self.tanh_(parent) * np.sqrt(len(tmp_length))))
                                num_bins = max(num_bins, self.traverse_threshold) 
                            else:
                                num_bins = int(self.bins)
                            if rm_outlier:
                                cutting_rm = self.outlier(Xdf[feature])
                                traverse = self.bin_method(cutting_rm, num_bins)
                            else:
                                traverse = self.bin_method(Xdf[feature], num_bins)
                        traverse = sorted(traverse)
              
                        for value in traverse:
                            left_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])
                            right_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                            left_y = Xdf[Xdf[feature] >= value]['Y']
                            right_y = Xdf[Xdf[feature] < value]['Y']
                            gain_temp = self.info_gain(parent, left_counts, right_counts, left_y, right_y)
                  
                            if best_feature is None:
                                best_feature = feature
                            if best_value is None:
                                best_value = value
                            if max_gain is None:
                                max_gain = gain_temp

                            max_gain, best_feature, best_value = self.trace_gain(max_gain, gain_temp, feature, value, self.info_method, best_feature, best_value) 
                                    
            traverse_copy = traverse
            traverse = None
        return (best_feature, best_value, traverse_copy, Xdf)

    def grow_tree(self):
        df = self.X.copy() 
        df['Y'] = self.Y
        # info_ma = self.gini_impur if self.info_method == 'gini' else self.variance
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            best_feature, best_value, traverse, df = self.best_split(df = df, outlier_ = self.outlier_)

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value

                if (isinstance(best_value, str)) | (type(best_value) == str):
                    left_df, right_df = df[df[best_feature]==best_value].copy(), df[df[best_feature] != best_value].copy()
                    rule_sign_left = '=='
                    rule_sign_right = '!='
                else:
                    left_df, right_df = df[df[best_feature]>=best_value].copy(), df[df[best_feature]<best_value].copy()
                    rule_sign_left = '>='
                    rule_sign_right = '<'

                left = Node(
                    left_df[self.features], 
                    left_df['Y'].values.tolist(), 
                    self.labels,
                    self.outlier_,
                    traverse_threshold = self.traverse_threshold,
                    min_samples_split=self.min_samples_split, 
                    max_depth=self.max_depth, 
                    node_type='left_node',
                    depth=self.depth + 1, 
                    na_threshold = self.na_threshold,
                    info_method = 'variance',
                    na_method = self.na_method,
                    bins = self.bins,
                    rule=f"{best_feature}" + rule_sign_left + f"{round(best_value, 3) if not isinstance(best_value, str) else best_value}")

                self.left = left 
                self.left.grow_tree()
                
                right = Node(
                    right_df[self.features], 
                    right_df['Y'].values.tolist(), 
                    self.labels,
                    self.outlier_,
                    traverse_threshold = self.traverse_threshold,
                    min_samples_split=self.min_samples_split, 
                    max_depth=self.max_depth, 
                    node_type='right_node',
                    depth=self.depth + 1, 
                    na_threshold = self.na_threshold,
                    info_method = 'variance',
                    na_method = self.na_method,
                    bins = self.bins,
                    rule=f"{best_feature}" + rule_sign_right + f"{round(best_value, 3) if not isinstance(best_value, str) else best_value}"
                    )

                self.right = right
                self.right.grow_tree()

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | Impurity information of the node: {round(self.gini_impur, 2) if self.info_method != 'variance' else round(self.variance, 2)}")
        print(f"{' ' * const}   | Class distribution in the node: {dict(self.counts)}")
        print(f"{' ' * const}   | Predicted class: {self.yhat}")   
  
    def print_tree(self):
        self.print_info() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()
  
    def predict(self, X:pd.DataFrame):
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_obs(values))
        
        return predictions

    def predict_obs(self, values: dict) -> int:
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            # Traversing the nodes all the way to the bottom
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 
      
            if (isinstance(values.get(best_feature), str)):
                if (values.get(best_feature) == best_value):
                    if self.left is not None:
                        cur_node = cur_node.left
                else:
                    if self.right is not None:
                        cur_node = cur_node.right

            else:
                if (values.get(best_feature) >= best_value):
                    if self.left is not None:
                        cur_node = cur_node.left

                else:
                    if self.right is not None:
                        cur_node = cur_node.right
            
        return cur_node.yhat
          