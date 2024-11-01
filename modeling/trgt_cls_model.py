import logging

import torch
from torch import nn

from config.features import ACTIVE_FEATURES, Privileged_FEATURES
from modeling.embedding import Embedding_Left, Embedding_Right
from modeling.feature_cross_net import DNN, TRGT, CrossNet
from modeling.pretrain_model import PretrainedModel
from modeling.transformer import GateTransformerEncoder
from utils.run import get_parameter_number

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


class TRTG_CLS_Model(nn.Module):
    def __init__(self, features=ACTIVE_FEATURES, trgt_layers=3, pretrained_weights_path=None, w_t=0.5, aux_loss_weight=0.5, device='cuda:0', **kwargs):
        super(TRTG_CLS_Model, self).__init__(**kwargs)
        self.aux_loss_weight = aux_loss_weight
        self.w_t = w_t
        self.device = device
        self.embedding_left = Embedding_Left(features, 4, 4, 16)
        self.embedding_right = Embedding_Right(Privileged_FEATURES, 4, 4)
        self.right_encoder = GateTransformerEncoder(1, 4, 2, 16)
        self.trgt = TRGT(trgt_layers, self.embedding_left.seq_len, self.embedding_left.d_model)
        self.pretrained_model = PretrainedModel(device=self.device)
        self.pretrained_model.load_weights(pretrained_weights_path)
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.hidden_dim = self.embedding_left.hidden_dim + self.pretrained_model.projection_dim
        self.mlp = DNN([self.hidden_dim, self.hidden_dim//2, self.hidden_dim//4, self.hidden_dim//8, 3])
        self.mlp_teacher = DNN([32+self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 4, self.hidden_dim // 8, 3])

        self.gate_outputs_dnn = nn.ModuleList([DNN([self.embedding_left.seq_len, 4]) for _ in range(trgt_layers)])


        self.ce = nn.CrossEntropyLoss()

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
        _, features, labels = self._preprocess(batch)
        flat, hidden, _ = self.embedding_left(features)
        output, gate_outputs = self.trgt(hidden)
        output = torch.flatten(output, start_dim=1, end_dim=-1)
        cls = self.pretrained_model.left_cls(features)
        logits = self.mlp(torch.cat((output, cls), dim=1))
        loss = self._ziln_loss(logits, labels['ltv3'])
        p, value = self._ziln_predict(logits)

        r_p = torch.flatten(self.right_encoder(self.embedding_right(features)[1]), start_dim=1, end_dim=-1)[:, :self.pretrained_model.projection_dim]
        logits_teacher = self.mlp_teacher(torch.cat((torch.flatten(output, start_dim=1, end_dim=-1), r_p), dim=1))
        loss += self.w_t * self._ziln_loss(logits_teacher, labels['ltv3'])

        # 分类label
        ltv_delta = torch.stack([torch.zeros_like(labels['ltv_3h'], device=labels['ltv_3h'].device), labels['ltv_3h'], labels['ltv_6h'] - labels['ltv_3h'], labels['ltv3'] - labels['ltv_6h']], dim=-1).clamp(min=0.)
        cls_label = torch.argmax(ltv_delta, dim=-1)

        for gate_output, gate_output_dnn in zip(gate_outputs, self.gate_outputs_dnn):
            cls_logits = gate_output_dnn(gate_output)
            loss += self.aux_loss_weight * self.ce(cls_logits, cls_label)

        return_dict = {
            'logit': logits,
            'loss': loss,
            'prediction_p': p,
            'prediction_v': value
        }
        return return_dict

    def _preprocess(self, batch):
        configs, features, labels = batch
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features[k] = v.to(self.device)
        for k, v in labels.items():
            if isinstance(v, torch.Tensor):
                labels[k] = v.to(self.device)
        return configs, features, labels

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