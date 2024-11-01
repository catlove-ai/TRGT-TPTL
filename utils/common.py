import json
import os
import logging
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from config.features import ALL_FEATURES, LOGIN_DENSE, DAPAN_DENSE, DAPAN_SEQ, get_feature_names, GAME_BEHAVIOR_DENSE, \
    TLOG_DENSE_30

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


def get_root_path():
    return r"D:/codes/www"


def read_csv_list(csv_list, columns=None, sep=',', max_rows=None):
    """
    :param sep:
    :param columns:
    :param csv_list: csv文件路径的list
    :return: 从多个csv读取df，按行合并
    """
    df_list = []
    rows = 0
    for csv_file in csv_list:
        new_df = pd.read_csv(csv_file, usecols=columns, sep=sep)
        logging.info(f'Read dataframe from csv {csv_file} | sample number: {len(new_df)}')
        df_list.append(new_df)
        rows += len(new_df)
        if max_rows is not None and rows >= max_rows:
            break
    df = pd.concat(df_list, axis=0, ignore_index=True)
    if max_rows is not None:
        df = df.head(max_rows)
    logging.info(f'sample num: {len(df)}')
    return df


def dense_boundary(jf_game_id, feat_delta, start_date, end_date, bucket_num=100, columns=None):
    """
    :param end_date:
    :param start_date:
    :param feat_delta:
    :param bucket_num: 分桶数量
    :param columns: 哪些列需要分桶，一般是dense特征列
    :return:
    """
    csv_list = [os.path.join(get_root_path(), f'data/{jf_game_id}/{jf_game_id}_{feat_delta}h_{date}.csv') for date in range(start_date, end_date+1)]
    csv_list = [csv for csv in csv_list if os.path.exists(csv)]

    df = read_csv_list(csv_list, columns)
    df.replace(0, np.nan, inplace=True)
    df_desc = df.describe()

    def cal_boundary(df_desc_line):
        def get_thresh(_min, _mid, _max):
            thresh0 = np.array([0])
            if _mid > _min:
                delta1 = (_mid - _min) / (bucket_num / 2 - 2)
                thresh1 = np.arange(_min, _mid + delta1 / 2, delta1)
            else:
                thresh1 = np.array([0] * int(bucket_num / 2 - 1))

            if _max > _mid:
                delta2 = (_max - _mid) / (bucket_num / 2 - 1)
                thresh2 = np.arange(_mid + delta2, _max + delta2 / 2, delta2)
            else:
                thresh2 = np.array([0] * int(bucket_num / 2 - 1))

            thresh = np.concatenate((thresh0, thresh1, thresh2), axis=0)
            return thresh

        # get the threshold for each node
        _max = df_desc_line.loc["max"]
        _mid = df_desc_line.loc['50%']
        _min = df_desc_line.loc['min']
        thr = get_thresh(_min, _mid, _max)
        return thr

    sr_boundary = df_desc.apply(cal_boundary)
    df_boundary = pd.DataFrame(sr_boundary.to_dict())
    save_path = os.path.join(get_root_path(), f'data/dense_boundaries/{jf_game_id}_dense_boundary_{feat_delta}h.csv')
    df_boundary.to_csv(save_path, index=False)
    logging.info(f"Saving bucket boundary to cache: {save_path}")
    return df_boundary


def seq_boundary(jf_game_id, feat_delta, start_date, end_date, bucket_num=10, columns=None):
    csv_list = [os.path.join(get_root_path(), f'data/{jf_game_id}/{jf_game_id}_{feat_delta}h_{date}.csv') for date in range(start_date, end_date + 1)]
    csv_list = [csv for csv in csv_list if os.path.exists(csv)]
    df = read_csv_list(csv_list, columns)
    buckets = {}
    for column in columns:
        all_numbers = []
        for seq_str in tqdm(df[column].tolist(), total=len(df), desc=column):
            seq = [float(x.strip()) for x in str(seq_str).split(',')]
            seq = [x for x in seq if x > 0]
            all_numbers += seq
        all_numbers.sort()
        per_l = [all_numbers[x * int(len(all_numbers) / bucket_num)] for x in range(1, bucket_num)]
        buckets[column] = per_l
    save_path = os.path.join(get_root_path(), f'data/seq_boundaries/{jf_game_id}_seq_boundary_{feat_delta}h.json')
    json.dump(buckets, open(save_path, 'w'), indent=4, ensure_ascii=False)
    return buckets


def cal_auc(labels, predictions):
    return roc_auc_score(labels, predictions)


def cal_gini(labels, predictions):
    df_model = pd.DataFrame({
        'y_true': labels,
        'y_pred': predictions,
    }).sort_values(by='y_pred', ascending=False)
    df_ground = pd.DataFrame({'y_true': labels}).sort_values(by='y_true', ascending=False)
    model_cum = (df_model['y_true'].cumsum() / df_model['y_true'].sum()).values
    ground_cum = (df_ground['y_true'].cumsum() / df_ground['y_true'].sum()).values
    model_gini = 2 * np.sum(model_cum) / model_cum.shape[0] - 1.
    ground_gini = 2 * np.sum(ground_cum) / ground_cum.shape[0] - 1.
    return model_gini / ground_gini


def cal_p_r(labels, predictions, top_rate):
    df = pd.DataFrame({
        'y_true': labels,
        'y_pred': predictions
    })
    prediction_sort = df.sort_values(by='y_pred', ascending=False).head(int(top_rate * len(df)))

    correct_num = (prediction_sort['y_true'] > 0).sum()
    p = correct_num / len(prediction_sort)
    r = correct_num / (df['y_true'] > 0).sum()
    return p, r


def cal_amount_recall_rate(labels, predictions, top_rate):
    df = pd.DataFrame({
        'y_true': labels,
        'y_pred': predictions
    })
    prediction_sort = df.sort_values(by='y_pred', ascending=False).head(int(top_rate * len(df)))
    ground_sort = df.sort_values(by='y_true', ascending=False).head(int(top_rate * len(df)))
    amount_sum = prediction_sort['y_true'].sum()
    base = ground_sort['y_true'].sum()
    return amount_sum, amount_sum / base


def RMSE(labels, preds):
    n = len(labels)
    guiyi = max(labels) - min(labels)
    labels = torch.tensor(labels, device='cuda:0')
    preds = torch.tensor(preds, device='cuda:0')
    rmse = torch.sqrt(torch.sum((labels-preds)**2) / n).detach().item()
    nrmse = rmse / guiyi
    return rmse, nrmse


def MAE(labels, preds):
    n = len(labels)
    guiyi = max(labels) - min(labels)
    labels = torch.tensor(labels, device='cuda:0')
    preds = torch.tensor(preds, device='cuda:0')
    rmse = (torch.sum(torch.abs(labels - preds)) / n).detach().item()
    nrmse = rmse / guiyi
    return rmse, nrmse


def evaluate(predict_df, save_path=None, prefix='default', label_col='ltv3'):
    eval_result = {}
    # auc
    auc = cal_auc(predict_df['is_pay'].values, predict_df['pay_probs'].values)
    eval_result['auc'] = auc
    # gini
    gini = cal_gini(predict_df[label_col], predict_df['p' + label_col])
    eval_result['gini_by_pltv'] = gini
    # gini
    gini = cal_gini(predict_df[label_col], predict_df['pay_probs'])
    eval_result['gini_by_p'] = gini
    # rmse
    eval_result['rmse'], eval_result['nrmse'] = RMSE(predict_df[label_col], predict_df['p' + label_col])
    # mae
    eval_result['mae'], eval_result['nmae'] = MAE(predict_df[label_col], predict_df['p' + label_col])
    # p,r
    for top_rate in [0.01, 0.03, 0.05, 0.1]:
        p, r = cal_p_r(predict_df[label_col], predict_df['p' + label_col], top_rate)
        eval_result[f'top_{top_rate}_p_by_pltv'] = p
        eval_result[f'top_{top_rate}_r_by_pltv'] = r
    for top_rate in [0.01, 0.03, 0.05, 0.1]:
        p, r = cal_p_r(predict_df[label_col], predict_df['pay_probs'], top_rate)
        eval_result[f'top_{top_rate}_p_by_p'] = p
        eval_result[f'top_{top_rate}_r_by_p'] = r

    # top amount recall rate
    eval_result['top_amount_recall_by_pltv'] = {}
    for top_rate in [0.01, 0.05, 0.1]:
        amount, rate = cal_amount_recall_rate(predict_df[label_col], predict_df['p' + label_col], top_rate)
        eval_result[f'top_{top_rate}_amount_recall_by_pltv'] = amount
    eval_result['top_amount_recall_by_p'] = {}
    for top_rate in [0.01, 0.05, 0.1]:
        amount, rate = cal_amount_recall_rate(predict_df[label_col], predict_df['pay_probs'], top_rate)
        eval_result[f'top_{top_rate}_amount_recall_by_p'] = amount

    if save_path is not None:
        json.dump(eval_result, open(os.path.join(save_path, f'{prefix}_eval_result.json'), 'w'), indent=4)

    return eval_result


if __name__ == '__main__':
    # dense_boundary('af', 6, 805, 825, columns=LOGIN_DENSE + DAPAN_DENSE + GAME_BEHAVIOR_DENSE + TLOG_DENSE_30)
    # seq_boundary('af', 6, 805, 825, columns=['onlinetime_seq', 'payment_seq'])

    dense_boundary('mg', 6, 805, 825, columns=LOGIN_DENSE + DAPAN_DENSE + GAME_BEHAVIOR_DENSE + TLOG_DENSE_30)
    seq_boundary('mg', 6, 805, 825, columns=['onlinetime_seq', 'payment_seq'])
