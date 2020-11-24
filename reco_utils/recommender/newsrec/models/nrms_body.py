# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn.functional as F
import torch.nn as nn

__all__ = ["NRMSModelBody"]

from reco_utils.recommender.newsrec.models.nrms import NRMSModel, AttLayer


class NRMSModelBody(NRMSModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.body_att_layer = AttLayer(hparams.word_emb_dim, hparams.attention_hidden_dim)
        self.body_self_att = nn.MultiheadAttention(hparams.word_emb_dim, hparams.head_num)

    def news_encoder(self, sequences_input):
        title, body = sequences_input[:, :self.hparams.title_size], sequences_input[:, self.hparams.title_size:]
        q = F.dropout(self.embedding_layer(title), p=self.hparams.dropout).transpose(0, 1)
        # Body embedding
        y = self.embedding_layer(body)
        y = y.reshape((body.shape[0], self.hparams.article_size, self.hparams.title_size, self.hparams.word_emb_dim))
        y = y.transpose(0, 1)
        y = torch.stack([self.news_att_layer(s) for s in y])
        y = y.transpose(0, 1)

        # Average embedding
        # y = torch.mean(y, dim=-2)
        # Body attention layer
        y = F.dropout(y, p=self.hparams.dropout).transpose(0, 1)
        y = self.body_self_att(q, y, y)[0].transpose(0, 1)
        # body_embedding = self.body_att_layer(body_embedding)
        # y = F.dropout(y, p=self.hparams.dropout).transpose(0, 1)
        # q = self.news_self_att(q, q, q)[0].transpose(0, 1)
        # y = torch.cat((q, y), 1)
        y = F.dropout(y, p=self.hparams.dropout)
        y = self.news_att_layer(y)
        return y



