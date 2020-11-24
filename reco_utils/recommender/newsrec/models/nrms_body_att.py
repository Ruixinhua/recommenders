# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn.functional as F
from torchnlp.nn.attention import Attention

__all__ = ["NRMSModelBodyAtt"]

from reco_utils.recommender.newsrec.models.nrms import NRMSModel


class NRMSModelBodyAtt(NRMSModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.attention = Attention(hparams.word_emb_dim)

    def multi_head_att(self, s):
        y = F.dropout(s, p=self.hparams.dropout).transpose(0, 1)
        y = self.news_self_att(y, y, y)[0].transpose(0, 1)
        return y

    def news_encoder(self, sequences_input):
        title, body = sequences_input[:, :self.hparams.title_size], sequences_input[:, self.hparams.title_size:]
        # title attention
        q = F.dropout(self.embedding_layer(title), p=self.hparams.dropout).transpose(0, 1)
        # shape of q: [N, S, E]     N: batch size, S: sequence length, E: embedded size
        q = self.news_self_att(q, q, q)[0].transpose(0, 1)
        # shape of q: [N, S, E]     E is head_num * head_dim
        q = F.dropout(q, p=self.hparams.dropout)
        q = self.news_att_layer(q)
        q = torch.unsqueeze(q, 1)
        # shape of q: [N, 1, E]

        # Body embedding
        # shape of y: [N, A*S, E]
        y = self.embedding_layer(body)

        # shape of y: [N, A*S, E]
        y = self.multi_head_att(y)
        y = y.reshape((body.shape[0], self.hparams.article_size, self.hparams.title_size, self.hparams.word_emb_dim))
        # shape of y: [N, A, S, E]      A: article size
        y = torch.stack([self.news_att_layer(F.dropout(s, p=self.hparams.dropout)) for s in y])

        # y = y.transpose(0, 1)
        # Article multi-head representation
        # shape of y: [A, N, S, E]
        # y = torch.stack([self.news_att_layer(F.dropout(self.multi_head_att(s), p=self.hparams.dropout))
        #                  for s in y]).transpose(0, 1)
        # use title as query and body as context.
        # shape of y: [N, A, E]
        y = self.attention(q, y)[0]
        y = torch.squeeze(y, 1)
        # shape of y: [N, E]
        return y



