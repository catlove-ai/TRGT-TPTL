import logging

import torch
from torch import nn

from config.features import get_feature_names, DAPAN_SPARSE, DAPAN_DENSE
from modeling.embedding import Embedding_Left
from modeling.feature_cross_net import DNN, MoE, DCN
from modeling.transformer import PositionEmbedding, AttentionPoolingLayer, EncoderLayer
from utils.run import get_parameter_number

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


class CMLTV_Model(nn.Module):
    def __init__(self, features, sparse_embedding_dim=4,
                 dense_embedding_dim=4, dense_embedding_dim_dapan=16, use_seq=True, device='cuda:0', **kwargs):
        super(CMLTV_Model, self).__init__(**kwargs)
        self.features = features
        self.use_seq = use_seq
        self.device = device

        self.embedding_all = Embedding_Left(features, sparse_embedding_dim, dense_embedding_dim, dense_embedding_dim_dapan)

        # hidden dim
        self.hidden_dim = self.embedding_all.hidden_dim

        self.ziln_dnn = DNN([self.hidden_dim, 3])
        self.l_dnn = DNN([self.hidden_dim, 1])
        # self.c_dnn = DNN([self.hidden_dim, 15])

        # device
        self.to(self.device)

        # log model info
        model_info = {
            'hidden_dim': self.hidden_dim,
            'params': get_parameter_number(self),
            'device': self.device
        }

        logging.info(f'base model info: {model_info}')

    def forward(self, batch):
        """
        :param batch:
        :return: batch loss
        """
        _, features, labels = self._preprocess(batch)

        flatten_hidden, _, _ = self.embedding_all(features)

        logits_ziln = self.ziln_dnn(flatten_hidden)
        logits_l = torch.relu(self.l_dnn(flatten_hidden)).squeeze()
        # logits_c = self.c_dnn(flatten_hidden)

        ziln_loss = self._ziln_loss(logits_ziln, labels)
        l_loss = torch.mean((logits_l-torch.log(1+labels))**2)

        p, v, _ = self._ziln_predict(logits_ziln)

        loss = ziln_loss + l_loss

        value = p * (0.5*v+0.5*l_loss)

        return_dict = {
            'logit': None,
            'loss': loss,
            'prediction_p': p,
            'prediction_v': value
        }
        return return_dict

    def _preprocess(self, batch):
        configs, features, labels = batch
        if isinstance(labels, dict):
            labels = labels['ltv3']
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features[k] = v.to(self.device)
        labels = labels.to(self.device)
        return configs, features, labels

    def _decode(self, hidden):
        output = self.cross_layer(hidden)
        return output

    @staticmethod
    def _ziln_loss(logits, label, label_clip_val=0., class_threshold=0.):
        if label_clip_val > 0:
            label = torch.clamp(label, min=0, max=label_clip_val)

        p_pred = logits[:, 0]
        p_ground = (label > class_threshold).to(torch.float32)

        cls_loss = nn.BCEWithLogitsLoss(reduction='none')(p_pred, p_ground)

        mu = logits[:, 1]
        sigma = torch.maximum(nn.Softplus()(logits[:, 2]), torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))

        safe_labels = p_ground * label + (1 - p_ground) * torch.ones_like(label)
        reg_loss = 0. - p_ground * torch.distributions.LogNormal(mu, sigma).log_prob(safe_labels)

        return torch.mean(cls_loss + reg_loss, dim=0)

    @staticmethod
    def _ziln_predict(logits, pred_clip_val=0):
        p, mu, sigma = logits[:, 0], logits[:, 1], logits[:, 2]
        p = torch.nn.Sigmoid()(p)
        sigma = torch.nn.Softplus()(sigma)

        v = torch.exp(mu + 0.5 * torch.square(sigma))
        preds = p * v



        # # 空值补0
        # is_nan = torch.isnan(preds)
        # padding_preds = torch.zeros_like(preds)
        # preds = torch.where(is_nan, padding_preds, preds)

        if pred_clip_val > 0:
            preds = torch.clamp(preds, min=0, max=pred_clip_val)

        return p, v, preds

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))