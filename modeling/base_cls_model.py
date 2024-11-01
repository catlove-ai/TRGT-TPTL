import logging

import torch
from torch import nn

from config.features import get_feature_names, DAPAN_SPARSE, DAPAN_DENSE
from modeling.embedding import Embedding_Left
from modeling.feature_cross_net import DNN, MoE, DCN
from modeling.pretrain_model import PretrainedModel
from modeling.transformer import PositionEmbedding, AttentionPoolingLayer, EncoderLayer
from utils.run import get_parameter_number

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


class Base_CLS_Model(nn.Module):
    def __init__(self, features, sparse_embedding_dim=4,
                 dense_embedding_dim=4, dense_embedding_dim_dapan=16, pretrained_weights_path=None, use_seq=True, w_t=0.5, cross_net='DNN', device='cuda:0', **kwargs):
        super(Base_CLS_Model, self).__init__(**kwargs)
        self.features = features
        self.use_seq = use_seq
        self.device = device
        self.w_t = w_t

        self.embedding_all = Embedding_Left(features, sparse_embedding_dim, dense_embedding_dim, dense_embedding_dim_dapan)

        # hidden dim
        self.hidden_dim = self.embedding_all.hidden_dim

        self.pretrained_model = PretrainedModel(device=self.device)
        self.pretrained_model.load_weights(pretrained_weights_path)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        output_dim = 3
        if cross_net == 'DNN':
            self.cross_layer = DNN([self.hidden_dim+32, 3])
        elif cross_net == 'MoE':
            self.cross_layer = MoE(2, [self.hidden_dim+32, 3])
        elif cross_net == 'DCN':
            self.cross_layer = DCN(1, [self.hidden_dim+32, 3], self.hidden_dim+32, 8, 3)
        else:
            raise NotImplementedError
        # self.cross_layer = MoELayer(num_experts=3, input_dim=self.hidden_dim, output_dim=output_dim)

        self.out_layer = DNN([64+32, 3])

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
        # hidden = self.cross_layer(flatten_hidden)
        cls = self.pretrained_model.left_cls(features)
        logits = self.cross_layer(torch.cat((flatten_hidden, cls), dim=1))
        loss = self._ziln_loss(logits, labels)
        p, value = self._ziln_predict(logits)
        return_dict = {
            'logit': logits,
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

        preds = (p * torch.exp(mu + 0.5 * torch.square(sigma)))

        # # 空值补0
        # is_nan = torch.isnan(preds)
        # padding_preds = torch.zeros_like(preds)
        # preds = torch.where(is_nan, padding_preds, preds)

        if pred_clip_val > 0:
            preds = torch.clamp(preds, min=0, max=pred_clip_val)

        return p, preds

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))