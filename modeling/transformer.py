import torch
import numpy as np
import torch.nn as nn


def activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    raise RuntimeError("activation should be relu/gelu/selu/leakyrelu, not {}".format(activation))


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def forward(self, x):
        length = x.shape[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding.
        x *= torch.sqrt(self.d_model).to(x.device)
        x = x + self.pos_encoding[torch.newaxis, :length, :]
        return x

    @staticmethod
    def positional_encoding(length, depth):
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
        angle_rates = 1 / (10000 ** depths)  # (1, depth)
        angle_rads = positions * angle_rates  # (pos, depth)
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return torch.tensor(pos_encoding, dtype=torch.float32)


class GlobalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.mha(query=x, value=x, key=x)
        x = x + attn_output[0]
        x = self.layer_norm(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.Dropout(dropout_rate),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.seq(x)  # ((20, 14) (20, 4))
        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model=4, num_heads=2, dff=4):
        super().__init__()
        self.self_attention = GlobalSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff) for _ in range(num_layers)]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        # Add dropout.
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape `(batch_size, seq_len, d_model)`.


class PositionEmbedding(nn.Module):
    """Implement position layer
    """

    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

    def forward(self, data):
        """forward pass"""
        max_len = data.shape[-2]
        positions = torch.arange(start=0, end=max_len, step=1, device=data.device)
        positions = self.pos_emb(positions)
        return data + positions


class AttentionPoolingLayer(nn.Module):

    def __init__(self, hidden_units, seq_length):
        super(AttentionPoolingLayer, self).__init__()
        self.hidden_units = hidden_units
        self.seq_length = seq_length
        self.att_fc1 = nn.Sequential(nn.Linear(2 * hidden_units, hidden_units), nn.ReLU())
        self.att_fc2 = nn.Linear(hidden_units, 1)

    def forward(self, queries, item_seq):
        weights = self.attention(queries, item_seq)
        seq_embedding = torch.matmul(weights, item_seq).view(-1, self.hidden_units)
        return seq_embedding

    def attention(self, q, k):
        """
        :param q: [bs,hidden_units]
        :param k: [bs,length,hidden_units]
        :return:
        """
        q = torch.cat([q] * self.seq_length, dim=1).view(-1, self.seq_length,
                                                         self.hidden_units)  # [bs, length, hidden_units]
        embed = torch.cat([q, k], dim=-1)  # [bs, length, 2*hidden_units]
        d_layer_1 = self.att_fc1(embed)
        d_layer_2 = self.att_fc2(d_layer_1).view(-1, 1, self.seq_length)
        outputs = d_layer_2 / (k.shape[-1] ** 0.5)
        weight = torch.softmax(outputs, dim=-1)

        return weight


class GateTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=16, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5, batch_first=True, use_layer_norm=True, norm_first=False, **kwargs):
        super(GateTransformerLayer, self).__init__(**kwargs)
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.use_layer_norm = use_layer_norm

        # attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=batch_first)

        # feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # gate
        self.gate_linear = nn.Linear(d_model, 1, bias=False)
        self.gate_act = nn.Sigmoid()

        if self.use_layer_norm:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation_fn(activation)

    def _sa_block(self, x, key_padding_mask):
        x = self.self_attn(query=x, value=x, key=x, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x):
        g = self.gate_act(self.gate_linear(x))  # GATE
        h = self.linear1(x)
        h = h * g
        h = self.linear2(self.dropout(self.activation(h)))
        return self.dropout2(h)

    def forward(self, src, src_mask=None):
        x = src
        if self.use_layer_norm:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask))
                x = self.norm2(x + self._ff_block(x))
        else:
            x = x + self._sa_block(x, src_mask)
            x = x + self._ff_block(x)
        return x


class GateTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=16, dropout=0.1, activation='relu',
                 layer_norm_eps=1e-5, batch_first=True, use_layer_norm=True, norm_first=False, **kwargs):
        super(GateTransformerEncoder, self).__init__(**kwargs)
        self.layers = nn.ModuleList([
            GateTransformerLayer(d_model=d_model,
                                 num_heads=num_heads,
                                 dim_feedforward=dim_feedforward,
                                 dropout=dropout,
                                 activation=activation,
                                 layer_norm_eps=layer_norm_eps,
                                 batch_first=batch_first,
                                 use_layer_norm=use_layer_norm,
                                 norm_first=norm_first,
                                 **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        x = src
        for layer in self.layers:
            x = layer(src, src_mask=src_mask)
        return x