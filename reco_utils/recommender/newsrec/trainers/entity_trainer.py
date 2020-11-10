# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import numpy as np
from reco_utils.recommender.newsrec.trainers.base_trainer import BaseTrainer

__all__ = ["EntityTrainer"]

from utils import tools


class EntityTrainer(BaseTrainer):

    def __init__(self, hparams, iterator_creator, seed=42):
        super().__init__(hparams, iterator_creator, seed)

    def _get_input_label_from_iter(self, batch_data):
        """ get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into trainer (clicked_title_batch & candidate_title_batch)
            array: labels
        """
        keys = ["clicked", "candidate"]
        input_feat = [
            torch.tensor(np.concatenate([batch_data[f"{k}_title_batch"],
                                        batch_data[f"{k}_entity_batch"]], axis=-1)).to(self.device)
            for k in keys
        ]
        input_label = batch_data["labels"]
        return input_feat, torch.tensor(input_label).to(self.device)

    def _get_user_feature_from_iter(self, batch_data):
        """ get input of user encoder

        Args:
            batch_data: input batch data from user iterator

        Returns:
            array: input user feature (clicked title batch)
        """
        return torch.tensor(np.concatenate([batch_data["clicked_title_batch"],
                                            batch_data["clicked_entity_batch"]], axis=-1)).to(self.device)

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder

        Args:
            batch_data: input batch data from news iterator

        Returns:
            array: input news feature (candidate title batch)
        """
        return torch.tensor(np.concatenate([batch_data["candidate_title_batch"],
                                            batch_data["candidate_entity_batch"]], axis=-1)).to(self.device)
