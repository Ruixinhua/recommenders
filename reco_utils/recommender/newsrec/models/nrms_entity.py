# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["NRMSModelEntity"]

from reco_utils.recommender.newsrec.models.nrms import NRMSModel, TimeDistributed


class NRMSModelEntity(NRMSModel):

    def __init__(self, hparams):
        super().__init__(hparams)

    def news_encoder(self, sequences_input):
        title, entity = sequences_input[:, :self.hparams.title_size], sequences_input[:, self.hparams.title_size:]
        y = F.dropout(self.embedding_layer(title), p=self.hparams.dropout).transpose(0, 1)
        # y = self.embedding_layer(title).transpose(0, 1)
        q = F.dropout(self.embedding_layer(entity), p=self.hparams.dropout).transpose(0, 1)
        # q = self.embedding_layer(entity).transpose(0, 1)
        y = self.news_self_att(q, y, y)[0].transpose(0, 1)
        y = F.dropout(y, p=self.hparams.dropout)
        y = self.news_att_layer(y)
        return y


