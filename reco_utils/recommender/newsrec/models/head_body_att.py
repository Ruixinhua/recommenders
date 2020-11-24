# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy

import torch
import torch.nn.functional as F
from torchnlp.nn.attention import Attention


__all__ = ["NRMSModelHeadBodyAtt"]

from reco_utils.recommender.newsrec.models.head_att import NRMSModelHeadAtt
from torchnlp.nn.attention import Attention

from utils import tools


class NRMSModelHeadBodyAtt(NRMSModelHeadAtt):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.head_attentions = self.clones(Attention(hparams.word_emb_dim // hparams.head_num), hparams.head_num)

    def reshape_head(self, y):
        return y.reshape(y.shape[0], self.hparams.head_num, -1).transpose(0, 1)

    def news_encoder(self, sequences_input):
        title, body = sequences_input[:, :self.hparams.title_size], sequences_input[:, self.hparams.title_size:]
        q = F.dropout(self.embedding_layer(title), p=self.hparams.dropout)
        # Body embedding
        y = F.dropout(self.embedding_layer(body), p=self.hparams.dropout)
        y = y.reshape((body.shape[0], self.hparams.article_size, self.hparams.title_size, self.hparams.word_emb_dim))
        # shape of y: [N, A, S, E]      A: article size
        y = y.transpose(0, 1)

        import time
        start = time.time()
        q = self.reshape_head(self.sentence_encoder(q))
        # shape of q: [N, H, D]      H: head number, D: head dimension
        y = torch.stack([self.reshape_head(self.sentence_encoder(s)) for s in y]).transpose(0, 1).transpose(1, 2)
        tools.print_log("Title and article cost:", time.time() - start, file=open(self.hparams.log_file, "a"))
        # shape of y: [H, N, A, D]      A: article size
        start = time.time()
        y = torch.stack([torch.squeeze(attention(torch.unsqueeze(t, 1), b)[0])
                         for t, b, attention in zip(q, y, self.head_attentions)])
        tools.print_log("Attention cost:", time.time() - start, file=open(self.hparams.log_file, "a"))
        # shape of y: [H, N, D]      A: article size
        y = y.transpose(0, 1)
        y = y.reshape(y.shape[0], -1)
        return y

