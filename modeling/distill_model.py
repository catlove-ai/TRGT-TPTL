import logging

import torch
from torch import nn

from config.features import ACTIVE_FEATURES, Privileged_FEATURES, ALL_FEATURES
from modeling.base_model import BaseModel
from modeling.embedding import Embedding_Left, Embedding_Right
from modeling.trtg_model import TRTGModel
from utils.run import get_parameter_number

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


class DistillModel(nn.Module):
    def __init__(self, teacher_weights_path, student_model='DNN', soft_loss_weight=0.5, use_seq=True, cross_net='DNN', device='cuda:0',
                 **kwargs):
        super(DistillModel, self).__init__(**kwargs)
        self.soft_loss_weight = soft_loss_weight
        self.use_seq = use_seq
        self.device = device

        self.teacher = BaseModel(features=ALL_FEATURES, device=device, cross_net=cross_net)
        self.teacher.load_weights(teacher_weights_path)
        for param in self.teacher.parameters():
            param.requires_grad = False
        if student_model == 'DNN':
            self.student = BaseModel(features=ACTIVE_FEATURES, device=device, cross_net=cross_net)
        elif student_model == 'TRGT':
            self.student = TRTGModel(features=ACTIVE_FEATURES, trgt_layers=1, device=device)

        # device
        self.to(self.device)

        # log model info
        model_info = {
            'teacher hidden_dim': self.teacher.hidden_dim,
            'student hidden_dim': self.student.hidden_dim,
            'params': get_parameter_number(self),
            'device': self.device
        }

        logging.info(f'base model info: {model_info}')

    def forward(self, batch):
        _, features, labels = self._preprocess(batch)
        teacher_output = self.teacher(batch)
        student_output = self.student(batch)
        hard_loss = student_output['loss']
        soft_loss = self._soft_ziln_loss(teacher_output['logit'], student_output['logit'])
        p, value = self._ziln_predict(student_output['logit'])

        return_dict = {
            'logit': student_output['logit'],
            'loss': hard_loss + self.soft_loss_weight * soft_loss,
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

    def _soft_ziln_loss(self, logits_t, logits_s, label_clip_val=0., class_threshold=0.):
        p, mu, sigma = nn.Sigmoid()(logits_t[..., 0]), logits_t[..., 1], logits_t[..., 2]
        soft_labels = (p * torch.exp(mu + 0.5 * torch.square(sigma)))
        if label_clip_val > 0:
            soft_labels = torch.clamp(soft_labels, min=0, max=label_clip_val)

        cls_loss = self._soft_cls_loss(p, nn.Sigmoid()(logits_s[..., 0]))
        reg_loss = self._soft_reg_loss(p, soft_labels, logits_s)

        return torch.mean(cls_loss + reg_loss, dim=0)

    @staticmethod
    def _soft_cls_loss(soft_probs, preds_probs):
        loss = (soft_probs - preds_probs) * (torch.log(soft_probs + 1e-10) - torch.log(preds_probs + 1e-10))
        return torch.mean(loss, dim=-1)

    @staticmethod
    def _soft_reg_loss(soft_probs, soft_labels, logits):
        mu = logits[..., 1]
        sigma = torch.maximum(nn.Softplus()(logits[..., 2]), torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))
        safe_labels = soft_labels + (1 - soft_probs) * torch.ones_like(soft_labels)
        reg_loss = 0. - torch.mean(soft_probs * torch.distributions.LogNormal(mu, sigma).log_prob(safe_labels), dim=-1)
        return reg_loss

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
