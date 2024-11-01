import torch
from torch import nn

from config.features import DAPAN_DENSE, get_feature_names, DAPAN_SPARSE
from modeling.transformer import AttentionPoolingLayer, EncoderLayer, PositionEmbedding


class Embedding_Left(nn.Module):
    def __init__(self, features, sparse_embedding_dim=4,
                 dense_embedding_dim=4, dense_embedding_dim_dapan=256, use_seq=True, seq_model='transformer', device='cuda:0'):
        super(Embedding_Left, self).__init__()
        self.features = features
        self.use_seq = use_seq
        self.device = device
        self.d_model = sparse_embedding_dim
        self.seq_model = seq_model

        # config features
        self.game_id_embedding_shared = nn.Embedding(1000, sparse_embedding_dim)  # gamie id embedding 序列特征使用
        self.media_id_embedding = nn.Embedding(1000, sparse_embedding_dim)
        self.media_type_embedding = nn.Embedding(100, sparse_embedding_dim)

        # sparse features
        self.sparse_embedding_dict = nn.ModuleDict({})
        self.sparse_embedding_dict_dapan = nn.ModuleDict({})
        for feature in self.features['sparse']:
            if feature.name in get_feature_names(DAPAN_SPARSE):
                self.sparse_embedding_dict_dapan.add_module(feature.name,
                                                            nn.Embedding(feature.vocab_size, sparse_embedding_dim))
                continue
            else:
                self.sparse_embedding_dict.add_module(feature.name,
                                                      nn.Embedding(feature.vocab_size, sparse_embedding_dim))

        # dense features
        self.dense_embedding_dict = nn.ModuleDict({})
        self.dense_embedding_dict_dapan = nn.ModuleDict({})
        for feature in self.features['dense']:
            if feature in DAPAN_DENSE:
                self.dense_embedding_dict_dapan.add_module(feature,
                                                           nn.Embedding(100, dense_embedding_dim_dapan))
                continue
            else:
                self.dense_embedding_dict.add_module(feature, nn.Embedding(100, dense_embedding_dim))

        attention_pooling_query_dim = (1 + len(DAPAN_SPARSE)) * sparse_embedding_dim

        # target_user_game_embedding维度变换,dapan
        self.dapan_pooling_query_convert = nn.Sequential(
            nn.Linear(attention_pooling_query_dim, dense_embedding_dim_dapan), nn.ReLU())
        # dapan dense features attention pooling
        self.dapan_pooling_layer = AttentionPoolingLayer(hidden_units=dense_embedding_dim_dapan,
                                                         seq_length=len(DAPAN_DENSE))

        # 序列特征
        if self.use_seq:
            self.seq_pooling_query_convert = nn.Sequential(nn.Linear(attention_pooling_query_dim, sparse_embedding_dim),
                                                           nn.ReLU())

            self.online_time_emb_layer = nn.Embedding(10, sparse_embedding_dim)
            self.payment_emb_layer = nn.Embedding(10, sparse_embedding_dim)
            transformer_layers = 1
            enc_layers = [EncoderLayer(d_model=self.d_model, num_heads=2, dff=4*self.d_model) for _ in range(transformer_layers)]
            self.enc_layers = nn.Sequential(*enc_layers)
            self.pos_emb_layer = PositionEmbedding(max_len=20, embed_dim=sparse_embedding_dim)
            self.seq_pooling_layer = AttentionPoolingLayer(hidden_units=sparse_embedding_dim, seq_length=20)

            # self.lstm = nn.LSTM(input_size=4, hidden_size=4, batch_first=True)

        self.hidden_dim = (3 + len(self.sparse_embedding_dict) + len(
            self.sparse_embedding_dict_dapan) + 3) * sparse_embedding_dim + len(
            self.dense_embedding_dict) * dense_embedding_dim + dense_embedding_dim_dapan

        self.seq_len = self.hidden_dim // sparse_embedding_dim

    def forward(self, inputs):
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
                if self.seq_model=='transformer':
                    # position encoding
                    x_first = self.pos_emb_layer(seq_embedding)
                    for i in range(transformer_layers):
                        x_first = self.enc_layers[i](x_first)
                    x_first = self.seq_pooling_layer(seq_query, x_first)
                    seq.append(x_first)
                else:
                    seq.append(self.lstm(seq_embedding)[1][0].squeeze())

        # 所有特征表示聚合
        feature_embedding_flat = torch.cat(
            [o2_game_id_hash, media_id_hash, media_type_hash] +
            sparse +
            sparse_dapan +
            dense +
            [dense_dapan_pooling] +
            seq, dim=1)
        # print(feature_embedding_flat.shape)
        return feature_embedding_flat, feature_embedding_flat.view(-1, self.seq_len, self.d_model), [
            torch.cat([o2_game_id_hash, media_id_hash, media_type_hash], dim=-1),
            torch.cat(sparse, dim=-1),
            torch.cat(sparse_dapan, dim=-1),
            torch.cat(dense, dim=-1),
            torch.cat([dense_dapan_pooling], dim=-1),
            torch.cat(seq, dim=-1)
        ]


class Embedding_Right(nn.Module):
    def __init__(self, features, sparse_embedding_dim=4,
                 dense_embedding_dim=4, device='cuda:0'):
        super(Embedding_Right, self).__init__()
        self.features = features
        self.device = device

        self.d_model = sparse_embedding_dim

        # sparse features
        self.sparse_embedding_dict = nn.ModuleDict({})
        for feature in self.features['sparse']:
            self.sparse_embedding_dict.add_module(feature.name, nn.Embedding(feature.vocab_size, sparse_embedding_dim))

        # dense features
        self.dense_embedding_dict = nn.ModuleDict({})
        for feature in self.features['dense']:
            self.dense_embedding_dict.add_module(feature, nn.Embedding(100, dense_embedding_dim))

        self.seq_len = len(self.sparse_embedding_dict) + len(self.dense_embedding_dict)
        self.hidden_dim = self.d_model * self.seq_len

    def forward(self, inputs):
        # sparse
        sparse = []
        for k, v in self.sparse_embedding_dict.items():
            sparse.append(v(inputs[k]))

        # dense
        dense = []
        for k, v in self.dense_embedding_dict.items():
            dense.append(v(inputs[k]))

        # 所有特征表示聚合
        feature_embedding_flat = torch.flatten(torch.stack(sparse + dense, dim=1), start_dim=1, end_dim=-1)
        # print(feature_embedding_flat.shape)

        return feature_embedding_flat, feature_embedding_flat.view(-1, self.seq_len, self.d_model), [torch.cat(sparse, dim=-1), torch.cat(dense, dim=-1)]


class New_Embedding(nn.Module):
    def __init__(self, features, embedding_dim=4, device='cuda:0'):
        super(New_Embedding, self).__init__()
        self.features = features
        self.device = device

        self.d_model = embedding_dim

        # sparse features
        self.sparse_embedding_dict = nn.ModuleDict({})
        for feature in self.features['sparse']:
            self.sparse_embedding_dict.add_module(feature.name,
                                                  nn.Embedding(feature.vocab_size, embedding_dim))

        # dense features
        self.dense_embedding_dict = nn.ModuleDict({})
        for feature in self.features['dense']:
            self.dense_embedding_dict.add_module(feature, nn.Embedding(100, embedding_dim))

        # seq
        self.seq_embedding_dict = nn.ModuleDict({})

        for feature in ['register_game_seq', 'active_game_seq', 'pay_game_seq']:
            if feature in get_feature_names(self.features['seq']):
                self.seq_embedding_dict.add_module(feature, nn.Embedding(1000, 4))

        for feature in ['onlinetime_seq', 'payment_seq']:
            if feature in get_feature_names(self.features['seq']):
                self.seq_embedding_dict.add_module(feature, nn.Embedding(10, 4))

        self.seq_len = len(self.sparse_embedding_dict) + len(self.dense_embedding_dict) + 20*len(self.seq_embedding_dict)

        self.hidden_dim = self.d_model * self.seq_len

    def forward(self, inputs):
        # sparse
        sparse = []
        for k, v in self.sparse_embedding_dict.items():
            sparse.append(v(inputs[k]).unsqueeze(1))

        # dense
        dense = []
        for k, v in self.dense_embedding_dict.items():
            dense.append(v(inputs[k]).unsqueeze(1))

        seq = []
        for k, v in self.seq_embedding_dict.items():
            seq.append(v(inputs[k]))


        # 所有特征表示聚合
        # feature_embedding_flat = torch.stack(sparse + dense, dim=1)

        feature_embedding_flat = torch.cat(sparse + dense + seq, dim=1)

        return feature_embedding_flat

