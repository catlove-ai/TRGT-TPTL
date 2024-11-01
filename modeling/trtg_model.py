import logging

import torch
from torch import nn

from config.features import get_feature_names, DAPAN_SPARSE, DAPAN_DENSE
from modeling.embedding import Embedding_Left, New_Embedding
from modeling.feature_cross_net import DNN, MoE, TRGT, CrossNet
from modeling.transformer import PositionEmbedding, AttentionPoolingLayer, EncoderLayer
from utils.run import get_parameter_number

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()]
                    )


class TRTGModel(nn.Module):
    def __init__(self, features, trgt_layers=3, sparse_embedding_dim=4,
                 dense_embedding_dim=4, dense_embedding_dim_dapan=16, wise='feature', use_seq=True, aux_loss_weight=0.5, device='cuda:0', **kwargs):
        super(TRTGModel, self).__init__(**kwargs)
        self.wise = wise
        self.features = features
        self.use_seq = use_seq
        self.device = device
        self.aux_loss_weight = aux_loss_weight

        self.d_model = sparse_embedding_dim

        self.embedding_all = Embedding_Left(features, sparse_embedding_dim, dense_embedding_dim, dense_embedding_dim_dapan)
        # self.embedding_all = New_Embedding(features, 4)

        # hidden dim
        self.hidden_dim = self.embedding_all.hidden_dim

        assert self.hidden_dim % sparse_embedding_dim == 0
        self.seq_len = self.hidden_dim // sparse_embedding_dim


        self.trgt = TRGT(trgt_layers, self.seq_len, sparse_embedding_dim, wise=wise)

        self.output_dnn = DNN([self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 4, 3])
        if wise=='feature':
            self.gate_outputs_dnn = nn.ModuleList([DNN([self.seq_len, 4]) for _ in range(trgt_layers)])
        elif wise=='channel':
            self.gate_outputs_dnn = nn.ModuleList([DNN([self.d_model, 4]) for _ in range(trgt_layers)])
        elif wise=='none':
            pass


        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

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

        flat, hidden, _ = self.embedding_all(features)
        output, gate_outputs = self.trgt(hidden)
        logits = self.output_dnn(torch.cat([torch.flatten(output, start_dim=1, end_dim=-1)], dim=1))
        loss = self._ziln_loss(logits, labels['ltv3'])
        p, value = self._ziln_predict(logits)

        if self.wise in ['feature', 'channel', 'none']:
            # 分类label
            ltv_delta = torch.stack([torch.zeros_like(labels['ltv_3h'], device=labels['ltv_3h'].device), labels['ltv_3h'], labels['ltv_6h']-labels['ltv_3h'], labels['ltv3']-labels['ltv_6h']], dim=-1).clamp(min=0.)
            cls_label = torch.argmax(ltv_delta, dim=-1)
            # 二分类
            # ltv_delta = torch.stack([labels['ltv_3h'], labels['ltv_6h'] - labels['ltv_3h'], labels['ltv3'] - labels['ltv_6h']], dim=-1).clamp(min=0.)
            # cls_label = torch.where(ltv_delta > 0., 1., 0.)

            for gate_output, gate_output_dnn in zip(gate_outputs, self.gate_outputs_dnn):
                cls_logits = gate_output_dnn(gate_output)
                loss += self.aux_loss_weight * self.ce(cls_logits, cls_label)

        return_dict = {
            'logit': logits,
            'loss': loss,
            'prediction_p': p,
            'prediction_v': value,
            'gat_outputs': gate_outputs,
            'cls_labels': cls_label,
            'trgt_output': torch.flatten(output, start_dim=1, end_dim=-1)
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

    def _encode(self, inputs):
        # config features
        o2_game_id_hash = self.game_id_embedding_shared(inputs['o2_game_id_hash'])
        media_type_hash = self.game_id_embedding_shared(inputs['media_type_hash'])
        media_id_hash = self.game_id_embedding_shared(inputs['media_id_hash'])

        # sparse
        sparse = []
        for k, v in self.sparse_embedding_dict.items():
            sparse.append(v(inputs[k]))
        sparse_dapan = []
        for k, v in self.sparse_embedding_dict_dapan.items():
            sparse_dapan.append(v(inputs[k]))

        # dense
        dense = []
        for k, v in self.dense_embedding_dict.items():
            dense.append(v(inputs[k]))
        dense_dapan = []
        for k, v in self.dense_embedding_dict_dapan.items():
            dense_dapan.append(v(inputs[k]))

        # pooling query
        pooling_query = torch.cat(sparse_dapan + [o2_game_id_hash], dim=-1)

        # dense dapan pooling
        dapan_query = self.dapan_pooling_query_convert(pooling_query)
        dense_dapan_pooling = self.dapan_pooling_layer(dapan_query, torch.stack(dense_dapan, dim=1))

        # seq
        seq = []
        if self.use_seq:
            online_time_emb = self.online_time_emb_layer(inputs['onlinetime_seq'])
            payment_emb = self.payment_emb_layer(inputs['payment_seq'])

            reg_game_emb = self.game_id_embedding_shared(inputs['register_game_seq'])
            act_game_emb = self.game_id_embedding_shared(inputs['active_game_seq'])
            pay_game_emb = self.game_id_embedding_shared(inputs['pay_game_seq'])
            act_emb = (act_game_emb + online_time_emb) / 2
            pay_emb = (pay_game_emb + payment_emb) / 2

            # 对三个序列进行transformer以及attention操作
            seq_query = self.seq_pooling_query_convert(pooling_query)
            seq_embeddings = [reg_game_emb, act_emb, pay_emb]
            transformer_layers = 1
            for seq_embedding in seq_embeddings:
                # position encoding
                x_first = self.pos_emb_layer(seq_embedding)
                for i in range(transformer_layers):
                    x_first = self.enc_layers[i](x_first)
                x_first = self.seq_pooling_layer(seq_query, x_first)
                seq.append(x_first)

        # 所有特征表示聚合
        feature_embedding_flat = torch.cat(
            [o2_game_id_hash, media_id_hash, media_type_hash] +
            sparse +
            sparse_dapan +
            dense +
            [dense_dapan_pooling] +
            seq, dim=-1)
        # print(feature_embedding_flat.shape)
        return feature_embedding_flat.view(-1, self.seq_len, self.d_model)

    def _decode(self, hidden):
        x = hidden.view(hidden.shape[0], -1, self.sparse_embedding_dim)
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

    def load_weights(self, path, strict=False):
        self.load_state_dict(torch.load(path), strict=strict)