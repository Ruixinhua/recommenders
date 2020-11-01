# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NRMSModel"]


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class AttLayer(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        # build attention network
        self.attention = nn.Sequential(
            nn.Linear(hparams.word_emb_dim, hparams.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(hparams.attention_hidden_dim, 1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        attention_weight = torch.unsqueeze(self.attention(x), 2)
        y = torch.sum(x * attention_weight, dim=1)
        return y


class NRMSModel(nn.Module):
    """NRMS trainer(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.array): Pretrained word embedding matrix.
        hparam (obj): Global hyper-parameters.
    """

    def __init__(self, hparams):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as head_num and head_dim are there.
        """
        super().__init__()
        self.hparams = hparams
        self.word2vec_embedding = np.load(hparams.wordEmb_file)
        self.embedding_layer = nn.Embedding(self.word2vec_embedding.shape[0], hparams.word_emb_dim).from_pretrained(
            torch.FloatTensor(self.word2vec_embedding), freeze=False)
        self.news_att_layer = AttLayer(hparams)
        self.user_att_layer = AttLayer(hparams)
        self.news_self_att = nn.MultiheadAttention(hparams.word_emb_dim, hparams.head_num)
        self.user_self_att = nn.MultiheadAttention(hparams.word_emb_dim, hparams.head_num)

    def news_encoder(self, sequences_input_title):
        embedded_sequences_title = self.embedding_layer(sequences_input_title)
        y = F.dropout(embedded_sequences_title, p=self.hparams.dropout).transpose(0, 1)
        y = self.news_self_att(y, y, y)[0]
        y = F.dropout(y, p=self.hparams.dropout).transpose(0, 1)
        y = self.news_att_layer(y)
        return y

    def user_encoder(self, his_input_title):
        # change size to (S, N, D): sequence length, batch size, word dimension
        y = TimeDistributed(self.news_encoder)(his_input_title).transpose(0, 1)
        # change size back to (N, S, D)
        y = self.user_self_att(y, y, y)[0].transpose(0, 1)
        y = self.user_att_layer(y)
        return y

    def forward(self, x):
        his_input_title, pred_input_title = x[0], x[1]
        user_present = self.user_encoder(his_input_title)
        news_present = TimeDistributed(self.news_encoder)(pred_input_title)
        # equal to Dot(axis=-1)([x, y])
        preds = torch.sum(news_present * user_present.unsqueeze(1), dim=-1)
        preds = F.softmax(preds, dim=-1)
        return preds

    def predict(self, x):
        hparams = self.hparams
        his_input_title, pred_input_title_one = x[0], x[1]
        user_present = self.user_encoder(his_input_title)
        pred_title_one_reshape = torch.reshape(pred_input_title_one, (hparams.title_size,))
        news_present_one = self.news_encoder(pred_title_one_reshape)
        # equal to Dot(axis=-1)([x, y])
        preds_one = torch.sum(news_present_one * user_present.unsqueeze(1), dim=-1)
        preds_one = F.sigmoid(preds_one)
        return preds_one
