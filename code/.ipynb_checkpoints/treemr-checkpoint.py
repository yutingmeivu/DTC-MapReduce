from collections import Counter
import math
import numpy as np
import math
import pandas as pd
import os


class Node_parallel:
    def __init__(self, X: list, Y: str, labels: list, features_df, cate_type, traverse_all, min_samples_split=None, max_depth=None,node_type=None, depth = None, info_method = 'variance', rule=None, method = None, num_bins = None, window = None, threshold = None):
        self.depth = depth if depth else 0
        self.X = X
        self.Y = Y
        self.features = [i for i in list(self.X[0].columns) if i != self.Y]
        self.counts = self.reduce_counter()
        self.labels = labels
        self.info_method = info_method
        self.n = sum(list(map(lambda x: x.shape[0], self.X)))
        self.cate_type = [None]
        self.max_gain = None
        # self.bins = bins if bins else math.floor(len(self.Y) / min_samples_split)
        self.num_bins = num_bins if num_bins else 20
        self.window = window if window else 2
        self.method = method if method else 'variance'
        # self.gini_impur = self.get_gini()
        # self.variance = self.get_var()
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 15
        self.node_type = node_type if node_type else 'root'
        self.rule = rule if rule else ""
        self.traverse_all = traverse_all
        self.threshold = threshold if threshold else 12
        self.impurity_info = self.get_gini_vari()
        self.features_df = features_df
        self.cate_type = cate_type
        self.traverse_all = traverse_all

        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]

        self.yhat = yhat 

        self.left = None 
        self.right = None 
        self.best_feature = None 
        self.best_value = None 
        self.max_gain = None

        # def partition_data(self, group_num, df):
        #   partitioned = [df.iloc[i:i+group_num] for i in range(0, len(df), group_num)]
        #   return partitioned

    def collect_info_partition(self, df, y, index):
        col_names = df[index].columns
        features = [i for i in col_names if i != y]
        return {col: df[index][col].value_counts().to_dict() for col in features}

    def collect_info_all(self, dict_collect):
        combined_dict = defaultdict(lambda: defaultdict(int))
        for dictionary in dict_collect:
            for key, value in dictionary.items():
                for sub_key, sub_value in value.items():
                  # Add the sub_value to the corresponding key in the combined_dict
                  combined_dict[key][sub_key] += sub_value
        result_dict = dict(combined_dict)
        return {key: dict(map(lambda x: (x[0], x[1]), value.items())) for key, value in result_dict.items()}

    def bin_method_syn(self, feature_dict, num_bins):
        values = sorted(list(feature_dict.keys()))
        bin_edges = np.linspace(start=min(values), stop=max(values), num=num_bins+1 if len(set(values)) > num_bins else (len(set(values)) + 1))
        hist, _ = np.histogram(values, bins=bin_edges)
        bdd =  _[1:-1]
        return bdd

    def ma_syn(self, feature_dict, window):
        value = sorted(list(feature_dict.keys()))
        return np.convolve(value, np.ones(window), 'valid') / window

    def get_datatype(self, df):
        cate_type = []
        for feature in df.columns:
            if df[feature].dtype == 'O':
                cate_type.append(feature)
        return cate_type

    def traverse_point(self, dict_all_collect, threshold, num_bins, window, method, cate_type, feature):
        traverse = {}
        if len(dict_all_collect[feature]) < threshold:
            traverse[feature] = list(dict_all_collect[feature].keys())
        else:
            # method: bin, ma(moving avg)
            # if feature not in self.cate_type:
            if method == 'bin':
                traverse[feature] = bin_method_syn(dict_all_collect[feature], num_bins)
            else:
                traverse[feature] = ma_syn(dict_all_collect[feature], window)
            # else:
            #   traverse[feature] = list(dict_all_collect[feature].keys())
        return traverse

    def get_feature(self, df, y):
        return [i for i in df.columns if i != y]

    def cat_to_num(self, labels, df, target):
        binary_codes = {labels[0]: 0, labels[1]: 1}
        y = df[target]
        encoded_values = [binary_codes[val] for val in y]
        return encoded_values

    @staticmethod
    def prob_compute(n, y1_count, y2_count):
        p1 = y1_count / n if n != 0 else 0
        p2 = y2_count / n if n != 0 else 0
        return p1, p2

    def reduce_counter(self):
        targets = [df[self.Y] for df in self.X]
        total_counts = Counter()
        for target in targets:
            total_counts.update(target)
        return total_counts

    def gini_impurity(self, n, y1_count, y2_count):
        # y1_count, y2_count: the # of corresponding label of y
        p1, p2 = self.prob_compute(n, y1_count, y2_count)
        gini = 1 - p1**2 - p2**2
        return gini
  
    @staticmethod
    def vari(n, s, q):
        var = 1/n * (q) - (s / n)**2 if n != 0 else 0
        return n * var

    def get_nsq(self, df_partitioned):
        d = df_partitioned
        n = 0 if len(d) is None else len(d)
        q = sum(list(map(lambda x: x**2, d)))
        s = sum(d)
        return (n, s, q)
  
    def get_rid_of_l(self, list_of_dicts):
        return {k: v for d in list_of_dicts for k, v in d.items()}

    def get_gini_n(self, df_partitioned):
        d = df_partitioned
        # can feed parent node, left child node, right child node
        y1_count, y2_count = Counter(d)[self.labels[0]], Counter(d)[self.labels[1]]
        total_count = y1_count + y2_count
        return (total_count, y1_count, y2_count)           
  
    def info_gain_combine(self, parent_info, left_info, right_info):
        if self.info_method == 'gini':
            gini_left = self.gini_impurity(left_info[0], left_info[1], left_info[2])
            gini_right = self.gini_impurity(right_info[0], right_info[1], right_info[2])
            gini_parent = self.gini_impurity(parent_info[0], parent_info[1], parent_info[2])

            n_left = left_info[0]
            n_right = right_info[0]

            w_left = n_left / (n_left + n_right)
            w_right = n_right / (n_left + n_right)

            wGINI = w_left * gini_left + w_right * gini_right

            gain = gini_parent - wGINI
        else:
            vari_left = self.vari(left_info[0], left_info[1], left_info[2])
            vari_right = self.vari(right_info[0], right_info[1], right_info[2])
            vari_parent = self.vari(parent_info[0], parent_info[1], parent_info[2])

            gain = vari_parent - (vari_left + vari_right)
      
        return gain

    def info_sync(self, parent, left_y, right_y):
        # y0_left, y1_left, y0_right, y1_right = left_counts.get(self.labels[0],0), left_counts.get(self.labels[1],0),\
        #  right_counts.get(self.labels[0],0), right_counts.get(self.labels[1],0)
        if self.info_method == 'variance':
            L_sub = self.get_nsq(left_y)
            R_sub = self.get_nsq(right_y)
            P_sub = self.get_nsq(parent)
        else:
            L_sub = self.get_gini_n(left_y)
            R_sub = self.get_gini_n(right_y)
            P_sub = self.get_gini_n(parent)
        feature_stat_sub = [P_sub, L_sub, R_sub]
        return feature_stat_sub # [(NP, SP, QP), (NL, SL, QL), (NR, SR, QR)] if variance else [(TP, 0P, 1P), (TL, 0L, 1L), (TR, 0R, 1R)]

    @staticmethod
    def trace_gain(max_gain, gain_temp, feature, value, info_method, best_feature, best_value):
        if gain_temp > max_gain:
            best_feature = feature
            best_value = value 
            max_gain = gain_temp
        return max_gain, best_feature, best_value

    def collect_parent_info_(self):
        if self.method == 'variance':
            info = self.get_nsq(self.X[self.Y])
        else:
            info = self.get_gini_n(self.X[self.Y])
        return info

    def reduce_current(self, collect_p):
        P_0 = sum(list(map(lambda x: x[0], list(collect_p))))
        P_1 = sum(list(map(lambda x: x[1], list(collect_p))))
        P_2 = sum(list(map(lambda x: x[2], list(collect_p))))
        return (P_0, P_1, P_2)

    def get_gini_vari(self):
        # pool = mp.Pool(20)
        collect = list(map(lambda x: self.get_nsq(x[self.Y]), self.X)) if self.method == 'variance' else  list(map(lambda x: self.get_gini_n(x[self.Y]), self.X))
        combined = self.reduce_current(collect)
        return self.gini_impurity(combined[0], combined[1], combined[2]) if self.method == 'gini' else self.vari(combined[0], combined[1], combined[2])
    

    def map_stat(self, value, feature, Xdf, id):
        sub_value = {}
        left_y = Xdf[id][Xdf[id][feature] >= value][self.Y]
        right_y = Xdf[id][Xdf[id][feature] < value][self.Y]
        sub_value[id] = self.info_sync(Xdf[id][self.Y], left_y, right_y) # {'id': [(P, P, P), (L, L, L), (R, R, R)]}
        return sub_value

    def map_seg_data(self, value, feature, Xdf, dtpe, id):
        # value: candidate cutting point from traverse_all
        # left_counts = Counter(Xdf[Xdf[feature] == value]['Y'])
        # right_counts = Counter(Xdf[Xdf[feature] != value]['Y'])
        # Xdf: d_p with formula [df1, df2, ..., dfn]
        if dtpe == 'string':
            left_y = Xdf[id][Xdf[id][feature] == value]
            right_y = Xdf[id][Xdf[id][feature] != value]
        else:
            left_y = Xdf[id][Xdf[id][feature] >= value]
            right_y = Xdf[id][Xdf[id][feature] < value]
        return (left_y, right_y)

    @staticmethod
    def sub_collect(sub_dict, index1, i):
        return sum(list(map(lambda x: x[index1][i], list(sub_dict.values()))))


    def reduce_value_stat(self, sub_dict_stat, value):
        # output_list = [(0, 0, 0)] * len(list(sub_dict_stat.values())[0])
        # for id_values in sub_dict_stat.values():
        #   for i, values in enumerate(id_values):
        #     output_list[i] = tuple(sum(x) for x in zip(output_list[i], values))

        P_0 = sum(list(map(lambda x: x[0][0], list(sub_dict_stat.values()))))
        P_1 = sum(list(map(lambda x: x[0][1], list(sub_dict_stat.values()))))
        P_2 = sum(list(map(lambda x: x[0][2], list(sub_dict_stat.values()))))
        L_0 = sum(list(map(lambda x: x[1][0], list(sub_dict_stat.values()))))
        L_1 = sum(list(map(lambda x: x[1][1], list(sub_dict_stat.values()))))
        L_2 = sum(list(map(lambda x: x[1][2], list(sub_dict_stat.values()))))
        R_0 = sum(list(map(lambda x: x[2][0], list(sub_dict_stat.values()))))
        R_1 = sum(list(map(lambda x: x[2][1], list(sub_dict_stat.values()))))
        R_2 = sum(list(map(lambda x: x[2][2], list(sub_dict_stat.values()))))
        return {value: [(P_0, P_1, P_2), (L_0, L_1, L_2), (R_0, R_1, R_2)]}


    def best_split(self):
        # measure: 'gini', 'variance'
        # features_copy = df.columns
        labels = self.labels
        info_method = self.info_method

        max_gain = None
        best_feature = None
        best_value = None
        Xdf = self.X.copy()

        # pool = mp.Pool(20)
        for feature in self.features:
            if Xdf is not None:
                for value in self.traverse_all[feature]:
                  # sync_stat_collect = pool.starmap_async(self.map_stat, [(value, feature, Xdf, id) for id in range(len(Xdf))]).get()
                    sync_stat_collect = list(map(lambda x: self.map_stat(value, feature, Xdf, x), range(len(Xdf))))
                    sync_stat_collect = self.get_rid_of_l(sync_stat_collect)
                    reduced = self.reduce_value_stat(sync_stat_collect, value) # {'value': [(Ps, Ps, Ps), (Ls, Ls, Ls), (Rs, Rs, Rs)]}

                    gain_async = self.info_gain_combine(reduced[value][0], reduced[value][1], reduced[value][2])

                    if best_feature is None:
                        best_feature = feature
                    if best_value is None:
                        best_value = value
                    if max_gain is None:
                        max_gain = gain_async
                    max_gain, best_feature, best_value = self.trace_gain(max_gain, gain_async, feature, value, self.info_method, best_feature, best_value)
                                  
        return (max_gain, best_feature, best_value)
  
    def segment_data(self, df, feature, value, id, cat):
        if cat:
            left_df, right_df = df[id][df[id][feature]==value], df[id][df[id][feature] != value]
        else:
            left_df, right_df = df[id][df[id][feature]>=value], df[id][df[id][feature] < value]
        return (left_df, right_df)

    def grow_tree(self):
        df = self.X.copy() 
        # pool = mp.Pool(20)
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            # print('not meet stopping criteria')
            max_gain, best_feature, best_value = self.best_split()
            # print('max_gain', max_gain, 'best_feature', best_feature, 'best_value', best_value)

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value
                self.max_gain = max_gain

                cat = False
                rule_sign_left = '>='
                rule_sign_right = '<'
              
                # df_seg = pool.starmap_async(self.map_seg_data, [(best_value, best_feature, df, cat, id) for id in range(len(df))]).get()
                df_seg = list(map(lambda x: self.map_seg_data(best_value, best_feature, df, cat, x), range(len(df))))
                left_df = list(map(lambda x: x[0], df_seg))
                right_df = list(map(lambda x: x[1], df_seg))
                left = Node_parallel(
                    left_df, 
                    self.Y, 
                    self.labels,
                    self.features_df, 
                    self.cate_type, 
                    self.traverse_all,
                    min_samples_split=self.min_samples_split, 
                    max_depth=self.max_depth, 
                    node_type='left_node',
                    depth=self.depth + 1, 
                    info_method = 'variance',
                    rule=f"{best_feature}" + rule_sign_left + f"{round(best_value, 3) if not isinstance(best_value, str) else best_value}",
                    method = 'bin', 
                    num_bins = 20, 
                    window = None,
                    threshold = 12
                    )

                self.left = left 
                self.left.grow_tree()
                
                right = Node_parallel(
                    right_df, 
                    self.Y, 
                    self.labels,
                    self.features_df, 
                    self.cate_type, 
                    self.traverse_all,
                    min_samples_split=self.min_samples_split, 
                    max_depth=self.max_depth, 
                    node_type='right_node',
                    depth=self.depth + 1, 
                    info_method = 'variance',
                    rule=f"{best_feature}" + rule_sign_right + f"{round(best_value, 3) if not isinstance(best_value, str) else best_value}",
                    method = 'bin', 
                    num_bins = 20, 
                    window = None,
                    threshold = 12
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
        print(f"{' ' * const}   | Impurity information gain of the node: {round(self.impurity_info, 2)}")
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
          