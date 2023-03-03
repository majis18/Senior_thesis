import numpy as np
import pandas as pd
import math

import umap

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Data_split():
    def __init__(self, df, per_train, per_valid, per_test, Y_label):
        self.df = df
        self.Y_label = Y_label
        self.n_row = df.shape[0]
        self.per_train = per_train
        self.per_valid = per_valid
        self.per_test = per_test
        self.n_train = 0
        self.n_valid = 0
        self.n_test = 0
        if self.per_train+self.per_valid+self.per_test==100:
            self.n_valid = math.floor(self.n_row * self.per_valid /100)
            self.n_test = math.floor(self.n_row * self.per_test / 100)
            self.n_train = self.n_row - self.n_valid - self.n_test
        else:
            print('Error: per_train, per_valid and per_test are not collected.')
    
    def split_for_classification(self):
        # 層化分割なしでtest, valid, trainに分割する
        # shuffled_df = self.df.sample(frac=1)
        # df_train = shuffled_df.iloc[0:self.n_train, :]
        # df_valid = shuffled_df.iloc[self.n_train:self.n_train+self.n_valid, :]
        # df_test = shuffled_df.iloc[self.n_train+self.n_valid:self.n_row, :]

        # train_oe_Y = pd.get_dummies(df_train[self.Y_label])
        # train_Y = torch.from_numpy(train_oe_Y.values).float().clone()
        # train_X = df_train.drop(self.Y_label, axis=1)
        # train_X = torch.from_numpy(train_X.T.values).float().clone()

        # valid_oe_Y = pd.get_dummies(df_valid[self.Y_label])
        # valid_Y = torch.from_numpy(valid_oe_Y.values).float().clone()
        # valid_X = torch.from_numpy(df_valid.drop(self.Y_label, axis=1).T.values).float().clone()

        # test_oe_Y = pd.get_dummies(df_test[self.Y_label])
        # test_Y = torch.from_numpy(test_oe_Y.values).float().clone()
        # test_X = torch.from_numpy(df_test.drop(self.Y_label, axis=1).T.values).float().clone()
        # return train_X, train_Y, valid_X, valid_Y, test_X, test_Y
        shuffled_df = self.df.sample(frac=1)
        df_Y = pd.get_dummies(shuffled_df[self.Y_label])
        df_X = shuffled_df.drop(self.Y_label, axis=1)

        train_Y = torch.from_numpy(df_Y.iloc[0:self.n_train, :].values).float().clone()
        train_X = torch.from_numpy(df_X.iloc[0:self.n_train, :].T.values).float().clone()
        valid_Y = torch.from_numpy(df_Y.iloc[self.n_train:self.n_train+self.n_valid, :].values).float().clone()
        valid_X = torch.from_numpy(df_X.iloc[self.n_train:self.n_train+self.n_valid, :].T.values).float().clone()
        test_Y = torch.from_numpy(df_Y.iloc[self.n_train+self.n_valid:self.n_row, :].values).float().clone()
        test_X = torch.from_numpy(df_X.iloc[self.n_train+self.n_valid:self.n_row, :].T.values).float().clone()
        return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

    def split_for_regression(self):
        # 層化分割なしでtest, valid, trainに分割する
        shuffled_df = self.df.sample(frac=1)
        df_train = shuffled_df.iloc[0:self.n_train, :]
        df_valid = shuffled_df.iloc[self.n_train:self.n_train+self.n_valid, :]
        df_test = shuffled_df.iloc[self.n_train+self.n_valid:self.n_row, :]

        train_Y = torch.from_numpy(df_train[self.Y_label].values).float().clone()
        train_X = df_train.drop(self.Y_label, axis=1)
        train_X = torch.from_numpy(train_X.T.values).float().clone()

        valid_Y = torch.from_numpy(df_valid[self.Y_label].values).float().clone()
        valid_X = torch.from_numpy(df_valid.drop(self.Y_label, axis=1).T.values).float().clone()

        test_Y = torch.from_numpy(df_test[self.Y_label].values).float().clone()
        test_X = torch.from_numpy(df_test.drop(self.Y_label, axis=1).T.values).float().clone()
        return train_X, train_Y, valid_X, valid_Y, test_X, test_Y



    def num_nodes(self):
        return self.df.shape[1]-1
    
    def num_classes(self):
        return self.df[self.Y_label].unique().shape[0]
    
    def real_per_print(self):
        print('train is {} %'.format(self.n_train/self.n_row))
        print('valid is {} %'.format(self.n_valid/self.n_row))
        print('test is {} %'.format(self.n_test/self.n_row))


# 次元削減手法UMAPを用いてグラフ構築
# k近傍グラフが出力されるはず　そのk値をいじりたい
class Graph_consutractor_umap():
    def __init__(self):
        self.um = umap.UMAP(random_state=0, transform_mode='graph')
    
    def fit(self, data):
        csr_graph = self.um.fit_transform(data)
        return csr_graph


class Transform_data():
    def __init__(self, data):
        self.data = data
    def generate_edge_index(self, mode):
        if mode == 'UMAP':
            um = Graph_consutractor_umap()
            um_g = um.fit(self.data)
            coo_g = um_g.tocoo()
            edge_index = torch.from_numpy(np.array([coo_g.row, coo_g.col])).clone().long()
            return edge_index


def generate_Data_List(x, y, edge_index):
    data_list = []
    for i in range(x.shape[1]):
        tmp_data = Data(x=x[:, i].unsqueeze(dim=1), edge_index=edge_index, y=y[i].unsqueeze(dim=0))
        data_list.append(tmp_data)
    return data_list


def generate_Data_Loader(data_list, batch_size, shuffle=False):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def set_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device