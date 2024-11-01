import os
import random
import logging

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.common import evaluate, get_root_path


# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
#                     handlers=[logging.StreamHandler()])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def plot_training(train_loss_step, val_loss_epoch, save_path, steps_per_epoch):
    plt.title('training process')

    x_train = range(len(train_loss_step))
    plt.plot(x_train, train_loss_step, color='red', label='training loss')

    if len(val_loss_epoch) > 0:
        x_val = [(i+1) * steps_per_epoch for i in range(len(val_loss_epoch))]
        x_val[-1] = min(x_val[-1], len(train_loss_step))
        plt.plot(x_val, val_loss_epoch, color='green', label='val loss', linestyle='--', marker='o')

    plt.legend()  # 显示图例
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, 'training_plot.png'))


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.epoch = 0
        self.best_epoch = 0
        self.best_loss = None
        self.delta = delta

    def __call__(self, val_loss, model):
        """
        :param val_loss:
        :param model:
        :return: early stop 状态， 0：无更新，1：存储当前model为best checkpoint，2：early stop
        """
        self.epoch += 1

        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            return 1
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
            else:
                return 0
        else:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.counter = 0
            return 1


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler=None,
                 early_stop: EarlyStopping = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stop = early_stop

        self.logger = logging.getLogger('training_logger')
        self.logger.setLevel(logging.INFO)

    def train(self, save_path, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader = None):
        # checkpoint path check
        if os.path.exists(save_path):
            raise FileExistsError()
        else:
            os.mkdir(save_path)

        file_handler = logging.FileHandler(os.path.join(save_path, 'training.log'))
        file_handler.setLevel(logging.DEBUG)

        # 创建一个Formatter，用于定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将FileHandler添加到日志记录器
        self.logger.addHandler(file_handler)

        training_loss_batch = []
        val_loss_epoch = []
        eval_result_list = {}
        for epoch in range(1, num_epochs + 1):
            # train one epoch
            self.model.train()
            train_loss_batch_mini = self._train_one_epoch(train_loader, epoch)
            training_loss_batch += train_loss_batch_mini

            # save checkpoint
            path = os.path.join(save_path, f'{epoch}.pth')
            self._save_checkpoint(path)

            if val_loader is not None:
                # validate
                self.model.eval()
                predict_df, val_loss = self.predict(val_loader)
                eval_result = evaluate(predict_df)
                eval_result_list[epoch] = eval_result
                self.logger.info(f'epoch {epoch} eval result: {eval_result}')

                # scheduler step
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                    self.logger.info(f'learning rate step to: {self.scheduler.get_last_lr()}')

        # plot training
        plot_training(training_loss_batch, val_loss_epoch, save_path, len(train_loader))

        pd.DataFrame(eval_result_list).T.to_csv(os.path.join(save_path, 'epoch_evaluation.csv'))

    def predict(self, test_loader, save_path=None, prefix='default', label_col='ltv3'):
        self.model.eval()
        # init
        result = {
            'user_type': [],
            'is_pay': [],
            'pay_probs': [],
            label_col: [],
            'p' + label_col: []
        }
        loss = []
        for batch in tqdm(test_loader, total=len(test_loader), desc='Predicting'):
            config, inputs, labels = batch
            if isinstance(labels, dict):
                labels = labels[label_col]
            # user_type
            result["user_type"] += inputs["user_type"].tolist()
            # ground
            result["is_pay"] += (labels > 0).to(torch.int32).tolist()
            result[label_col] += labels.tolist()
            # predict
            predict_result = self.model(batch)
            result["pay_probs"] += predict_result['prediction_p'].squeeze().tolist()
            result['p' + label_col] += predict_result['prediction_v'].squeeze().tolist()
            loss.append(predict_result['loss'].detach().item())

        result_df = pd.DataFrame(result)
        if save_path is not None:
            save_file = os.path.join(save_path, f'{prefix}_prediction.csv')
            result_df.to_csv(save_file, index=False)
            self.logger.info('Prediction saved to {}'.format(save_file))

        return result_df, np.mean(loss).item()

    def _train_one_epoch(self, train_loader: DataLoader, epoch_num: int, training_loss_window=20):
        train_loss_batch = []
        window_loss_batch = []
        loop_batch = tqdm(train_loader, total=len(train_loader), desc=f'Training of epoch {epoch_num}', leave=True)
        error_batches = 0
        for batch in loop_batch:

            loss = self.model(batch)['loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            training_loss = loss.detach().item()
            train_loss_batch.append(training_loss)
            window = min(len(train_loss_batch), training_loss_window)
            window_training_loss = np.mean(train_loss_batch[-window:])
            loop_batch.set_postfix(training_loss="{:.5f}".format(window_training_loss))
            window_loss_batch.append(window_training_loss)

        return window_loss_batch

    def _cal_val_loss(self, val_loader: DataLoader, epoch_num: int):
        val_loss_batch = []
        for batch in tqdm(val_loader, total=len(val_loader), desc=f'Evaluating of epoch {epoch_num}'):
            with torch.no_grad():
                loss = self.model(batch)['loss']
            val_loss_batch.append(loss.detach().item())
        return np.mean(val_loss_batch).item()

    def _save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
