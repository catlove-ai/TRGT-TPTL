import logging

import torch
from torch import nn
from torch.nn.functional import normalize

from config.features import ACTIVE_FEATURES, Privileged_FEATURES
from modeling.embedding import Embedding_Left, Embedding_Right
from modeling.transformer import GateTransformerEncoder, GateTransformerLayer
from utils.run import get_parameter_number

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


class PretrainedModel(nn.Module):
    def __init__(self, left_features=ACTIVE_FEATURES, right_features=Privileged_FEATURES, num_layers=3, loss_weights=[1, 0.5, 0.2], d_model: int = 8, num_heads: int = 2,
                 dim_feedforward: int = 16,
                 num_cls: int = 8, projection_dim: int = 32, temperature: float = 0.2,
                 device='cuda:0', **kwargs):
        """
        :param left_features: dict like: {'dense': [str], 'sparse':[dense feature object]}  大盘+激活特征
        :param right_features: dict like: {'dense': [str], 'sparse':[dense feature object]}   tlog特征
        :param kwargs:
        """
        super(PretrainedModel, self).__init__(**kwargs)
        self.num_cls = num_cls
        self.temperature = temperature
        self.device = device
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.loss_weights = loss_weights

        # embedding
        self.left_embedding = Embedding_Left(left_features, sparse_embedding_dim=d_model, dense_embedding_dim=d_model, dense_embedding_dim_dapan=4*d_model)
        self.right_embedding = Embedding_Right(right_features, sparse_embedding_dim=d_model, dense_embedding_dim=d_model)

        # encoder
        self.left_encoder = GateTransformerEncoder(num_layers, d_model, num_heads, dim_feedforward)
        self.right_encoders = nn.ModuleList([GateTransformerLayer(d_model, num_heads, dim_feedforward=dim_feedforward) for _ in range(num_layers)])

        # projection
        self.left_projection = nn.Linear(d_model * num_cls, projection_dim)
        self.right_projections = nn.ModuleList([nn.Linear(d_model * num_cls, projection_dim) for _ in range(num_layers)])

        self.ce = nn.CrossEntropyLoss()

        self.to(device)

        # log model info
        model_info = {
            'cls_dim': self.projection_dim,
            'params': get_parameter_number(self),
            'device': self.device
        }

        logging.info(f'base model info: {model_info}')

    def forward(self, batch):
        inputs = batch
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        left_x = self._left_forward(inputs)  # (batch_size, projection_dim)
        right_xs = self._right_forward(inputs)  # (batch_size, projection_dim)

        # 这里貌似用Norm layer还是normalize都可以
        # left_x = self.left_logits_norm(left_x)
        # right_x = self.right_logits_norm(right_x)
        left_x = normalize(left_x, dim=1)
        right_xs = [normalize(right_x, dim=1) for right_x in right_xs]

        return_dict = {
            'loss': sum([weight*self._contrastive_loss(left_x, right_x) for weight, right_x in zip(self.loss_weights, right_xs)])
            # 'loss': self._contrastive_loss(left_x, right_xs[-1])
        }

        return return_dict

    def _projection(self, x, which: str, right_number=0):
        batch_size = x.shape[0]
        x = x[:, :self.num_cls, :].view(batch_size, -1)
        if which == 'left':
            return self.left_projection(x)
        elif which == 'right':
            return self.right_projections[right_number](x)
        else:
            raise ValueError("Unknown projection")

    def _left_forward(self, inputs):
        _, x, _ = self.left_embedding(inputs)
        x = self.left_encoder(x)
        x = self._projection(x, 'left')
        return x

    def _right_forward(self, inputs):
        _, x, _ = self.right_embedding(inputs)
        right_outputs = []
        for i in range(self.num_layers):
            x = self.right_encoders[i](x)
            right_outputs.append(self._projection(x, which='right', right_number=i))
        return right_outputs

    def left_cls(self, inputs):
        inputs = inputs
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return self._left_forward(inputs)

    def _contrastive_loss(self, q, k):
        """
        info nce loss
        :return:
        """
        batch_size = q.shape[0]
        preds = torch.div(torch.matmul(q, k.T), self.temperature)
        labels = torch.arange(batch_size, device=q.device, dtype=torch.long)
        return self.ce(preds, labels)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

