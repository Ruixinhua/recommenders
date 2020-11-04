# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.backends import cudnn

from reco_utils.recommender.deeprec.deeprec_utils import cal_metric
from tqdm import tqdm

__all__ = ["BaseTrainer"]

from utils import tools


class CategoricalLoss(nn.Module):

    def __init__(self):
        super(CategoricalLoss, self).__init__()

    def forward(self, predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """
        predictions, targets = predictions.float(), targets.float()
        predictions = torch.clamp(predictions, epsilon, 1. - epsilon)
        return -torch.sum(targets * torch.log(predictions + 1e-9)) / predictions.shape[0]


class BaseTrainer:
    """
    Basic class of models
    """

    def __init__(self, hparams, iterator_creator, seed=42):
        """Initializing the trainer. Create common logics which are needed by all deeprec models, such as loss function,
        parameter set.

        Args:
            hparams (obj): A HParams object, hold the entire set of hyperparameters.
            iterator_creator_train (obj): An iterator to load the data in trainning steps.
            iterator_creator_train (obj): An iterator to load the data in testing steps.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.train_iterator = iterator_creator(hparams, hparams.npratio, col_spliter="\t")
        self.test_iterator = iterator_creator(hparams, col_spliter="\t")

        self.hparams = hparams
        self.support_quick_scoring = hparams.support_quick_scoring
        self.log_file = open(hparams.log_file, "a")
        model_type = f"{hparams.model_type}_{hparams.trainer}"
        self.model = tools.get_model_class(model_type, **{"hparams": hparams}).to(tools.get_device())
        self.best_model = self.model

        self.loss = self._get_loss()
        self.train_optimizer = opt.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        # set GPU use with demand growth
        cudnn.benchmark = False
        cudnn.deterministic = True

    def _get_input_label_from_iter(self, batch_data):
        """ get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into trainer (clicked_title_batch & candidate_title_batch)
            array: labels
        """
        input_feat = [
            torch.tensor(batch_data["clicked_title_batch"]).to(tools.get_device()),
            torch.tensor(batch_data["candidate_title_batch"]).to(tools.get_device()),
        ]
        input_label = batch_data["labels"]
        return input_feat, torch.tensor(input_label).to(tools.get_device())

    def _get_user_feature_from_iter(self, batch_data):
        """ get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            array: input user feature (clicked title batch)
        """
        return torch.tensor(batch_data["clicked_title_batch"]).to(tools.get_device())

    def _get_news_feature_from_iter(self, batch_data):
        """ get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            array: input news feature (candidate title batch)
        """
        return torch.tensor(batch_data["candidate_title_batch"]).to(tools.get_device())

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss

        Returns:
            obj: Loss function or loss function name
        """
        # TODO: modify loss function
        if self.hparams.loss == "cross_entropy_loss":
            # categorical_crossentropy in tensorflow1`
            # data_loss = nn.CrossEntropyLoss()
            data_loss = CategoricalLoss()
        elif self.hparams.loss == "log_loss":
            # binary_crossentropy in tensorflow
            data_loss = nn.BCELoss()
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.

        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)

        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = nn.Identity()(logit)
        elif task == "classification":
            pred = nn.Sigmoid()(logit)
        else:
            raise ValueError("method must be regression or classification, but now is {0}".format(task))
        return pred

    def train(self, train_batch_data):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The trainer session object.
            feed_dict (dict): Feed values to train the trainer. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        # TODO: modify train code
        train_input, train_label = self._get_input_label_from_iter(train_batch_data)
        self.model.train()
        output = self.model(train_input)
        # _, train_label = train_label.max(dim=1)
        loss = self.loss(output, train_label)
        self.train_optimizer.zero_grad()
        loss.backward()
        self.train_optimizer.step()
        return loss.item()

    def fit(self, train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, test_news_file=None,
            test_behaviors_file=None, model_dir="trainer", iter_step=10000, resume_path=None):
        """Fit the trainer with train_file. Evaluate the trainer on valid_file per epoch to observe the training status.
        If test_news_file is not None, evaluate it too.

        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_news_file (str): test set.

        Returns:
            obj: An instance of self.
        """

        best_score = 0.0
        if resume_path:
            self.model.load_state_dict(torch.load(resume_path))
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch
            epoch_loss = 0
            train_start = time.time()

            batch_data = self.train_iterator.load_data_from_file(train_news_file, train_behaviors_file)
            with tqdm(batch_data, file=sys.stdout) as bar:
                i = 0
                for batch_data_input in batch_data:
                    bar.set_description(f"Training {i+1}")
                    bar.update(1)
                    step_result = self.train(batch_data_input)
                    step_data_loss = step_result

                    epoch_loss += step_data_loss
                    step += 1

                    if step % iter_step == 0:
                        eval_res = self.run_eval(valid_news_file, valid_behaviors_file)
                        info = [f"{item[0]}:{item[1]}" for item in sorted(eval_res.items(), key=lambda x: x[0])]
                        eval_info = ", ".join(info)
                        tools.print_log(f"at epoch {epoch} at step {step}\neval info: {eval_info}", file=self.log_file)
                        cur_score = eval_res["group_auc"]
                        if cur_score > best_score:
                            best_score = cur_score
                            model_path = os.path.join(model_dir, f"ckpt_{epoch}_{step}.pth")
                            self.best_model = self.model
                            torch.save(self.model.state_dict(), os.path.join(model_dir, f"best_model.pth"))
                            torch.save(self.model.state_dict(), model_path)
                            tools.print_log(f"save model state at {model_path}", file=self.log_file)
                    i += 1

            tools.print_log(f"Epoch no: {epoch}, batch no: {step}.", file=self.log_file )

            train_end = time.time()
            train_time = train_end - train_start
            eval_start = time.time()

            train_info = ",".join([str(item[0]) + ":" + str(item[1]) for item in [("logloss loss", epoch_loss / step)]])
            eval_res = self.run_eval(valid_news_file, valid_behaviors_file)
            cur_score = eval_res["group_auc"]
            if cur_score > best_score:
                best_score = cur_score
                self.best_model = self.model
                torch.save(self.model.state_dict(), os.path.join(model_dir, f"best_model.pth"))
                tools.print_log(f"save best model", file=self.log_file)
            info = [str(item[0]) + ":" + str(item[1]) for item in sorted(eval_res.items(), key=lambda x: x[0])]
            eval_info = ", ".join(info)

            if test_news_file is not None:
                test_res = self.run_eval(test_news_file, test_behaviors_file)
                info = [str(item[0]) + ":" + str(item[1]) for item in sorted(test_res.items(), key=lambda x: x[0])]
                test_info = ", ".join(info)
                log = f"at epoch {epoch}\ntrain info: {train_info}\neval info: {eval_info}\ntest info: {test_info}"
                tools.print_log(log, file=self.log_file)
            else:
                tools.print_log(f"at epoch {epoch}\ntrain info: {train_info}\neval info: {eval_info}", file=self.log_file)
            eval_end = time.time()
            eval_time = eval_end - eval_start

            tools.print_log(f"at epoch {epoch} , train time: {train_time} eval time: {eval_time}", file=self.log_file)

        return self

    def eval(self, eval_batch_data):
        """Evaluate the data in feed_dict with current trainer.

        Args:
            sess (obj): The trainer session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        # TODO: modify evaluation code
        eval_input, eval_label = self._get_input_label_from_iter(eval_batch_data)
        imp_index = eval_batch_data["impression_index_batch"]
        pred_result = self.model.predict(eval_input)

        return pred_result, eval_label, imp_index

    def run_eval(self, news_filename, behaviors_file):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """
        self.model.eval()
        with torch.no_grad():
            if self.support_quick_scoring:
                _, group_labels, group_preds = self.run_fast_eval(news_filename, behaviors_file)
            else:
                _, group_labels, group_preds = self.run_slow_eval(news_filename, behaviors_file)
            res = cal_metric(group_labels, group_preds, self.hparams.metrics)
        return res

    def user(self, batch_user_input):
        user_input = self._get_user_feature_from_iter(batch_user_input)
        user_vec = self.model.user_encoder(user_input).cpu().numpy()
        user_index = batch_user_input["impression_index_batch"]
        return user_index, user_vec

    def news(self, batch_news_input):
        news_input = self._get_news_feature_from_iter(batch_news_input)
        news_vec = self.model.news_encoder(news_input).cpu().numpy()
        news_index = batch_news_input["news_index_batch"]
        return news_index, news_vec

    def run_user(self, news_filename, behaviors_file, test_set=False):
        if not hasattr(self.model, "user_encoder"):
            raise ValueError("trainer must have attribute user_encoder")

        user_indexes = []
        user_vecs = []
        batch_data = self.test_iterator.load_user_from_file(news_filename, behaviors_file, test_set=test_set)
        with tqdm(batch_data, file=sys.stdout) as bar:
            i = 0
            for batch_data_input in batch_data:
                bar.set_description(f"Run user {i+1}")
                bar.update(1)
                user_index, user_vec = self.user(batch_data_input)
                user_indexes.extend(np.reshape(user_index, -1))
                user_vecs.extend(user_vec)
                i += 1

        return dict(zip(user_indexes, user_vecs))

    def run_news(self, news_filename):
        if not hasattr(self.model, "news_encoder"):
            raise ValueError("trainer must have attribute news_encoder")

        news_indexes = []
        news_vecs = []
        batch_data = self.test_iterator.load_news_from_file(news_filename)
        with tqdm(batch_data, file=sys.stdout) as bar:
            i = 0
            for batch_data_input in batch_data:
                bar.set_description(f"Run news {i+1}")
                bar.update(1)
                news_index, news_vec = self.news(batch_data_input)
                news_indexes.extend(np.reshape(news_index, -1))
                news_vecs.extend(news_vec)
                i += 1
        return dict(zip(news_indexes, news_vecs))

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.

        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.

        Returns:
            all_labels: labels after group.
            all_preds: preds after group.

        """

        all_keys = list(set(group_keys))
        all_keys.sort()
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}

        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)

        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])

        return all_keys, all_labels, all_preds

    def run_slow_eval(self, news_filename, behaviors_file):
        preds = []
        labels = []
        imp_indexes = []

        batch_data = self.test_iterator.load_data_from_file(news_filename, behaviors_file)
        with tqdm(batch_data, file=sys.stdout) as bar:
            i = 0
            for batch_data_input in bar:
                bar.set_description(f"Run slow evaluation {i+1}")
                bar.update(1)
                step_pred, step_labels, step_imp_index = self.eval(batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                imp_indexes.extend(np.reshape(step_imp_index, -1))
                i += 1

        group_impr_indexes, group_labels, group_preds = self.group_labels(labels, preds, imp_indexes)
        return group_impr_indexes, group_labels, group_preds

    def run_fast_eval(self, news_filename, behaviors_file, test_set=False):
        news_vecs = self.run_news(news_filename)
        user_vecs = self.run_user(news_filename, behaviors_file, test_set=test_set)

        group_impr_indexes = []
        group_labels = []
        group_preds = []

        batch_data = self.test_iterator.load_impression_from_file(behaviors_file, test_set=test_set)
        with tqdm(batch_data, file=sys.stdout) as bar:
            i = 0
            for batch_data_input in batch_data:
                if test_set:
                    impr_index, news_index, user_index = batch_data_input
                else:
                    impr_index, news_index, user_index, label = batch_data_input
                bar.set_description(f"Run fast evaluation {i+1}")
                bar.update(1)
                pred = np.dot(np.stack([news_vecs[i] for i in news_index], axis=0), user_vecs[impr_index])
                group_impr_indexes.append(impr_index)
                if not test_set:
                    group_labels.append(label)
                group_preds.append(pred)
                i += 1
        return (group_impr_indexes, group_preds) if test_set else (group_impr_indexes, group_labels, group_preds)
