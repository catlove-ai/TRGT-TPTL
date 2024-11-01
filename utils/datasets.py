import os
import json
import hashlib
import logging

import torch
import pandas as pd

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils.common import read_csv_list, get_root_path
from config.features import get_feature_names, DAPAN_SEQ, ALL_FEATURES

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


def consistent_hash(input_string, bucket_num=100, seed=42):
    # 创建一个sha256的哈希对象
    hash_object = hashlib.sha256()

    # 更新哈希对象，包括输入字符串和种子值
    hash_object.update((input_string + str(seed)).encode('utf-8'))

    # 返回哈希值的十六进制表示
    return int(hash_object.hexdigest(), 16) % bucket_num  # 取模以限制哈希桶数量


def dense_bucket(x, thresholds):
    return sum([int(x > threshold) for threshold in thresholds])


def seq_bucket(x_str, thresholds):
    x_seq = [float(x) for x in x_str.split(',')]
    bucket_x_seq = [dense_bucket(x, thresholds) for x in x_seq]
    return bucket_x_seq


def label_cast(x, label_max_cast):
    if x < 0:
        return 0
    elif x > label_max_cast:
        return label_max_cast
    else:
        return x


class BaseDataset(Dataset):
    def __init__(self, jf_game_id, start_date, end_date, features, label_col='ltv3', minus_label_col=None,
                 label_max_cast=2000, max_rows=None):
        self.jf_game_id = jf_game_id
        self.features = features
        self.minus_label_col = minus_label_col
        self.label_max_cast = label_max_cast
        self.max_rows = max_rows
        self.features = features

        self.config_col = get_feature_names(features['config'])
        self.features_col = (features['dense'] + get_feature_names(features['sparse'])
                             + get_feature_names(features['seq'])
                             + ['o2_game_id_hash', 'media_type_hash', 'media_id_hash'])
        self.label_col = label_col

        csv_list = [os.path.join(get_root_path(), 'processed_data', f'{jf_game_id}/{jf_game_id}_6h_{date}.csv') for date in
                    range(start_date, end_date + 1)]
        csv_list = [csv for csv in csv_list if os.path.exists(csv)]

        columns = (get_feature_names(features['config']) + features['dense'] + get_feature_names(features['sparse'])
                   + get_feature_names(features['seq']) + [label_col])
        if minus_label_col is not None:
            columns.append(minus_label_col)

        df = read_csv_list(csv_list, columns, max_rows=max_rows)

        self.config_df, self.features_df, self.label_df = self._preprocess(df)

    def __len__(self):
        return len(self.config_df)

    def __getitem__(self, index: int):
        return self.config_df.iloc[index].to_dict(), self.features_df.iloc[index].to_dict(), self.label_df[index]

    def _preprocess(self, df):
        for col in ['register_game_seq', 'pay_game_seq', 'active_game_seq', 'onlinetime_seq', 'payment_seq']:
            df[col] = df[col].apply(lambda x: [int(i.strip()) for i in x.split(',')]).tolist()

        # label
        if self.minus_label_col is not None:
            df[self.label_col] = ((df[self.label_col] - df[self.minus_label_col]).apply(lambda x: label_cast(x, self.label_max_cast)))

        return df[self.config_col], df[self.features_col], df[self.label_col]


class MultiLabelDataset(Dataset):
    def __init__(self, jf_game_id, start_date, end_date, features, minus_label_col=None,
                 label_max_cast=2000, max_rows=None):
        self.jf_game_id = jf_game_id
        self.features = features
        self.minus_label_col = minus_label_col
        self.label_max_cast = label_max_cast
        self.max_rows = max_rows
        self.features = features

        self.config_col = get_feature_names(features['config'])
        self.features_col = (features['dense'] + get_feature_names(features['sparse'])
                             + get_feature_names(features['seq'])
                             + ['o2_game_id_hash', 'media_type_hash', 'media_id_hash'])
        self.label_col = get_feature_names(features['labels'])

        csv_list = [os.path.join(get_root_path(), 'processed_data', f'{jf_game_id}/{jf_game_id}_6h_{date}.csv') for date in
                    range(start_date, end_date + 1)]
        csv_list = [csv for csv in csv_list if os.path.exists(csv)]

        columns = (get_feature_names(features['config']) + features['dense'] + get_feature_names(features['sparse'])
                   + get_feature_names(features['seq']) + self.label_col)
        if minus_label_col is not None:
            columns.append(minus_label_col)

        df = read_csv_list(csv_list, columns, max_rows=max_rows)

        self.config_df, self.features_df, self.label_df = self._preprocess(df)

    def __len__(self):
        return len(self.config_df)

    def __getitem__(self, index: int):
        return self.config_df.iloc[index].to_dict(), self.features_df.iloc[index].to_dict(), self.label_df.iloc[index].to_dict()

    def _preprocess(self, df):
        for col in ['register_game_seq', 'pay_game_seq', 'active_game_seq', 'onlinetime_seq', 'payment_seq']:
            df[col] = df[col].apply(lambda x: [int(i.strip()) for i in x.split(',')]).tolist()

        # label
        if self.minus_label_col is not None:
            df[self.label_col] = ((df[self.label_col] - df[self.minus_label_col]).apply(lambda x: label_cast(x, self.label_max_cast)))

        return df[self.config_col], df[self.features_col], df[self.label_col]


class PretrainDataset(Dataset):
    def __init__(self, jf_game_id, start_date, end_date, features=ALL_FEATURES, max_rows=None):
        self.jf_game_id = jf_game_id
        self.features = features
        self.max_rows = max_rows
        self.features = features

        self.features_col = (features['dense'] + get_feature_names(features['sparse'])
                             + get_feature_names(features['seq'])
                             + ['o2_game_id_hash', 'media_type_hash', 'media_id_hash'])

        csv_list = [os.path.join(get_root_path(), 'processed_data', f'{jf_game_id}/{jf_game_id}_6h_{date}.csv') for date in
                    range(start_date, end_date + 1)]
        csv_list = [csv for csv in csv_list if os.path.exists(csv)]

        columns = (get_feature_names(features['config']) + features['dense'] + get_feature_names(features['sparse'])
                   + get_feature_names(features['seq']))

        df = read_csv_list(csv_list, columns, max_rows=max_rows)

        self.features_df = self._preprocess(df)

    def __len__(self):
        return len(self.features_df)

    def __getitem__(self, index: int):
        return self.features_df.iloc[index].to_dict()

    def _preprocess(self, df):
        for col in ['register_game_seq', 'pay_game_seq', 'active_game_seq', 'onlinetime_seq', 'payment_seq']:
            df[col] = df[col].apply(lambda x: [int(i.strip()) for i in x.split(',')]).tolist()

        return df[self.features_col]


def base_collate(batch):
    configs = {}
    features = {}
    config_names = batch[0][0].keys()
    feature_names = batch[0][1].keys()
    for name in config_names:
        configs[name] = [item[0][name] for item in batch]
    for name in feature_names:
        if name in get_feature_names(DAPAN_SEQ):
            features[name] = torch.tensor(
                [item[1][name] for item in batch])
        else:
            features[name] = torch.tensor([item[1][name] for item in batch])
    labels = torch.tensor([item[2] for item in batch])
    return configs, features, labels


def multi_label_collate(batch):
    configs = {}
    features = {}
    labels = {}
    config_names = batch[0][0].keys()
    feature_names = batch[0][1].keys()
    label_names = batch[0][2].keys()
    for name in config_names:
        configs[name] = [item[0][name] for item in batch]
    for name in feature_names:
        if name in get_feature_names(DAPAN_SEQ):
            features[name] = torch.tensor(
                [item[1][name] for item in batch])
        else:
            features[name] = torch.tensor([item[1][name] for item in batch])
    for name in label_names:
        labels[name] = torch.tensor([item[2][name] for item in batch])
    return configs, features, labels


def pretrain_collate(batch):
    """
    :param batch: list[feature_dict]
    :return:
    """
    features = {}
    feature_names = batch[0].keys()
    for name in feature_names:
        features[name] = torch.tensor([item[name] for item in batch])
    return features


if __name__ == '__main__':
    pass
