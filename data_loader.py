# data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from config import COLUMNS_TO_READ

class DataLoader:
    def __init__(self, file_path, train_ratio=0.6, val_ratio=0.4):
        """
        初始化数据加载器。

        参数：
        - file_path: 数据文件的路径。
        - train_ratio: 训练集所占比例（针对每个 profile_id）。
        - val_ratio: 验证集所占比例（针对每个 profile_id）。
        """
        self.file_path = file_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.load_data()
        self.split_data()

    def load_data(self):
        """从文件加载数据，只读取指定的列。"""
        # 仅读取 config.py 中指定的列
        self.data = pd.read_csv(self.file_path, usecols=COLUMNS_TO_READ)

    def split_data(self):
        """按新规则划分数据集。"""
        data = self.data

        # 选择需要按 6:4 划分的 profile_id 区间
        data_2_20_41_60 = data[((data['profile_id'] >= 2) & (data['profile_id'] <= 20)) |
                               ((data['profile_id'] >= 45) & (data['profile_id'] <= 65))]

        # 选择实时测试集的数据 (不在上述区间内的 profile_id)
        self.test_data = data[~(((data['profile_id'] >= 2) & (data['profile_id'] <= 20)) |
                                ((data['profile_id'] >= 45) & (data['profile_id'] <= 65)))].reset_index(drop=True)

        # 按每个 profile_id 划分训练集和验证集
        train_data_list = []
        val_data_list = []

        for pid, group in data_2_20_41_60.groupby('profile_id'):
            n = len(group)
            split_index = int(self.train_ratio * n)  # 按 6:4 划分
            train_data_list.append(group.iloc[:split_index])
            val_data_list.append(group.iloc[split_index:])

        # 合并训练集和验证集
        self.train_data = pd.concat(train_data_list).reset_index(drop=True)
        self.val_data = pd.concat(val_data_list).reset_index(drop=True)

    def get_train_data(self):
        """获取训练集数据。"""
        return self.train_data

    def get_val_data(self):
        """获取验证集数据。"""
        return self.val_data

    def get_test_data(self):
        """获取测试集数据。"""
        return self.test_data

    def get_all_data(self):
        """获取完整的数据集。"""
        return self.data

    def get_features_and_labels(self, dataset):
        """将数据集拆分为特征和标签。

        参数：
        - dataset: 数据集（训练集、验证集或测试集）。
        """
        X = dataset.drop(columns=['torque', 'profile_id'])  # 特征
        y = dataset['torque']  # 标签
        return X, y

    def get_train_features_and_labels(self):
        """获取训练集的特征和标签。"""
        return self.get_features_and_labels(self.train_data)

    def get_val_features_and_labels(self):
        """获取验证集的特征和标签。"""
        return self.get_features_and_labels(self.val_data)

    def get_test_features_and_labels(self):
        """获取测试集的特征和标签。"""
        return self.get_features_and_labels(self.test_data)
