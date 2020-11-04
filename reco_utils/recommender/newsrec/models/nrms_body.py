# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import torch.nn.functional as F

__all__ = ["NRMSModelBody"]

from reco_utils.recommender.newsrec.models.nrms import NRMSModel, TimeDistributed


class NRMSModelBody(NRMSModel):

    def __init__(self, hparams):
        super().__init__(hparams)

    def news_encoder(self, sequences_input):
        title, body = sequences_input[:, :self.hparams.title_size], sequences_input[:, self.hparams.title_size:]
        q = F.dropout(self.embedding_layer(title), p=self.hparams.dropout).transpose(0, 1)
        body_embedding = self.embedding_layer(body)
        body_embedding = body_embedding.reshape((body.shape[0], self.hparams.article_size, self.hparams.title_size,
                                                 self.hparams.word_emb_dim))
        y = F.dropout(torch.mean(body_embedding, dim=-2), p=self.hparams.dropout).transpose(0, 1)
        y = self.news_self_att(q, y, y)[0].transpose(0, 1)
        y = F.dropout(y, p=self.hparams.dropout)
        y = self.news_att_layer(y)
        return y

    def embedding(self, input_content):
        title_input, body_input = input_content
        title_embedding = F.dropout(self.embedding_layer(title_input), p=self.hparams.dropout)
        body_embedding = F.dropout(self.embedding_layer(body_input), p=self.hparams.dropout)
        body_embedding = torch.mean(body_embedding, dim=-2)
        return torch.cat((title_embedding, body_embedding), dim=-2)


