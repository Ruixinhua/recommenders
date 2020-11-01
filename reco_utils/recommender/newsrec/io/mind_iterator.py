# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import json

import numpy as np
import pickle

from reco_utils.recommender.newsrec.newsrec_utils import word_tokenize, newsample

__all__ = ["MINDIterator"]


class MINDIterator(object):
    """Train data loader for NAML trainer.
    The trainer require a special type of data format, where each instance contains a label, impresion id, user id,
    the candidate news articles and user's clicked news article. Articles are represented by title words,
    body words, verts and subverts. 

    Iterator will not load the whole data into memory. Instead, it loads data into memory
    per mini-batch, so that large files can be used as input data.

    Attributes:
        col_spliter (str): column spliter in one line.
        ID_spliter (str): ID spliter in one line.
        batch_size (int): the samples num in one batch.
        title_size (int): max word num in news title.
        his_size (int): max clicked news num in user click history.
        npratio (int): negaive and positive ratio used in negative sampling. -1 means no need of negtive sampling.
    """

    def __init__(self, hparams, npratio=-1, col_spliter="\t", ID_spliter="%", ):
        """Initialize an iterator. Create necessary placeholders for the trainer.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key setyings such as head_num and head_dim are there.
            npratio (int): negative and positive ratio used in negative sampling. -1 means no need of negative sampling.
            col_spliter (str): column splitter in one line.
            ID_spliter (str): ID splitter in one line.
        """
        self.col_spliter = col_spliter
        self.ID_spliter = ID_spliter
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.title_size = hparams.title_size
        self.his_size = hparams.his_size
        self.npratio = npratio

        self.word_dict = self.load_dict(hparams.wordDict_file)
        self.uid2index = self.load_dict(hparams.userDict_file)

        # initial news and behaviours data
        self.news = {attr: [""] for attr in hparams.news_attr.keys()}

    @staticmethod
    def load_dict(file_path):
        """ load pickle file
        Args:
            file path (str): file path
        
        Returns:
            (obj): pickle load obj
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def init_matrix(self, data, shape):
        matrix = np.zeros(shape, dtype="int32")
        for index in range(matrix.shape[0]):
            content = data[index]
            for word_index in range(min(len(content), matrix.shape[1])):
                if content[word_index] in self.word_dict:
                    matrix[index, word_index] = self.word_dict[content[word_index].lower()]
        return matrix

    @staticmethod
    def load_entity(entity):
        return " ".join([" ".join(e["SurfaceForms"]) for e in json.loads(entity)])

    def _init_news(self, news_file):
        """ init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        """

        self.nid2index = {}

        with open(news_file, "r", encoding="utf-8") as rd:
            for line in rd:
                # news id, category, subcategory, title, abstract, url
                nid, vert, subvert, title, ab, url, title_entity, abs_entity = line.strip("\n").split(self.col_spliter)
                entities = self.load_entity(title_entity) if len(title_entity) > 2 else self.load_entity(abs_entity)
                entities = entities if len(entities) else title
                news_dict = {"title": title, "entity": entities, "vert": vert, "subvert": subvert, "abstract": ab}
                if nid in self.nid2index:
                    continue
                for attr in self.news.keys():
                    if attr in news_dict:
                        self.news[attr].append(word_tokenize(news_dict[attr]))
                self.nid2index[nid] = len(self.nid2index) + 1

        self.news_index_matrix = {
            f"news_{attr}_index": self.init_matrix(self.news[attr],   # news data
                                                   (len(self.news[attr]), self.hparams.news_attr[attr]))  # matrix shape
            for attr in self.news.keys()
        }
        self.news_title_index = self.init_matrix(self.news["title"], (len(self.news["title"]), self.title_size))
        # self.news_entity_index = self.init_matrix(self.news["entity"], (len(self.news["entity"]), self.title_size))

    def _init_behaviors(self, behaviors_file, test_set=False):
        """ init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        self.histories = []
        self.imprs = []
        self.labels = []
        self.impr_indexes = []
        self.uindexes = []

        with open(behaviors_file, "r", encoding="utf-8") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(self.col_spliter)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.his_size - len(history)) + history[:self.his_size]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                if not test_set:
                    label = [int(i.split("-")[1]) for i in impr.split()]
                    self.labels.append(label)
                uindex = self.uid2index[uid] if uid in self.uid2index else 0

                self.histories.append(history)
                self.imprs.append(impr_news)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1

    def parser_one_line(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negtive sampled result.
        
        Args:
            line (int): sample index.

        Returns:
            list: Parsed results including label, impression id , user id, 
            candidate_title_index, clicked_title_index.
        """
        if self.npratio > 0:
            impr_label = self.labels[line]
            impr = self.imprs[line]

            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                # divide click and un-click news
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                label = [1] + [0] * self.npratio
                impr_index = [self.impr_indexes[line]]
                user_index = [self.uindexes[line]]
                # random shuffle training data
                # state = np.random.get_state()
                # np.random.shuffle(label)
                impression = {"labels": label, "impression_index_batch": impr_index, "user_index_batch": user_index}
                for attr in self.news_index_matrix.keys():
                    candidate_index = self.news_index_matrix[attr][[p] + newsample(negs, self.npratio)]
                    clicked_index = self.news_index_matrix[attr][self.histories[line]]
                    attr = attr.replace("news_", "").replace("index", "batch")
                    # keep the same random state as label shuffle
                    # np.random.set_state(state)
                    # np.random.shuffle(candidate_index)
                    impression.update({f"candidate_{attr}": candidate_index, f"clicked_{attr}": clicked_index})
                yield impression

        else:
            # use all impression news as training data
            impr_label = self.labels[line]
            impr = self.imprs[line]

            for news, label in zip(impr, impr_label):
                label = [label]
                impr_index = [self.impr_indexes[line]]
                user_index = [self.uindexes[line]]
                impression = {"labels": label, "impression_index_batch": impr_index, "user_index_batch": user_index}
                for attr in self.news_index_matrix.keys():
                    candidate_index = [self.news_index_matrix[attr][news]]
                    clicked_index = self.news_index_matrix[attr][self.histories[line]]
                    attr = attr.replace("news_", "").replace("index", "batch")
                    impression.update({f"candidate_{attr}": candidate_index, f"clicked_{attr}": clicked_index})
                yield impression

    def load_data_from_file(self, news_file, behavior_file):
        """Read and parse data from news file and behavior file.
        
        Args:
            news_file (str): A file contains several informations of news.
            behavior_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed results, in the format of dict.
        """

        if not hasattr(self, "news_index_matrix"):
            self._init_news(news_file)

        if not hasattr(self, "imp_indexes"):
            self._init_behaviors(behavior_file)

        keys = ["labels", "impression_index_batch", "user_index_batch"]
        for attr in self.news_index_matrix.keys():
            attr = attr.replace("news_", "").replace("index", "batch")
            keys.extend([f"candidate_{attr}", f"clicked_{attr}"])
        impression_dict = {key: [] for key in keys}
        cnt = 0

        indexes = np.arange(len(self.labels))

        if self.npratio > 0:
            np.random.shuffle(indexes)

        for i in indexes:
            for impression in self.parser_one_line(i):
                for key in impression_dict.keys():
                    impression_dict[key].append(impression[key])
                cnt += 1
                if cnt >= self.batch_size:
                    yield self._convert_data(impression_dict)
                    impression_dict = {key: [] for key in keys}
                    cnt = 0

        if cnt > 0:
            yield self._convert_data(impression_dict)

    @staticmethod
    def _convert_data(data_dict):
        """Convert data into numpy arrays that are good for further trainer operation.
        
        Args:
            data_dict: a dictionary with multiple list
        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        types_mapping = {"labels": np.float32, "impression_index_batch": np.int32,
                         "user_index_batch": np.int32, "news_index_batch": np.int32}
        for key in data_dict.keys():
            dtype = types_mapping[key] if key in types_mapping else np.int64
            data_dict[key] = np.asarray(data_dict[key], dtype=dtype)
        return data_dict

    def load_user_from_file(self, news_file, behavior_file, test_set=False):
        """Read and parse user data from news file and behavior file.
        
        Args:
            news_file (str): A file contains several informations of news.
            behavior_file (str): A file contains information of user impressions.

        Returns:
            obj: An iterator that will yields parsed user feature, in the format of dict.
        """

        if not hasattr(self, "news_index_matrix"):
            self._init_news(news_file)

        if not hasattr(self, "imp_indexes"):
            self._init_behaviors(behavior_file, test_set=test_set)

        keys = ["impression_index_batch", "user_index_batch"]
        for attr in self.news_index_matrix.keys():
            keys.extend([f"clicked_{attr.replace('news_', '').replace('index', 'batch')}"])
        users_dict = {key: [] for key in keys}
        cnt = 0

        for index in range(len(self.impr_indexes)):
            user = {"user_index_batch": self.uindexes[index], "impression_index_batch": self.impr_indexes[index]}
            for attr, news_index in self.news_index_matrix.items():
                attr = attr.replace("news_", "").replace("index", "batch")
                user.update({f"clicked_{attr}": news_index[self.histories[index]]})
            for attr in user:
                users_dict[attr].append(user[attr])

            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_data(users_dict)
                users_dict = {key: [] for key in keys}
                cnt = 0

        if cnt > 0:
            yield self._convert_data(users_dict)

    def load_news_from_file(self, news_file):
        """Read and parse user data from news file.
        
        Args:
            news_file (str): A file contains several information of news.
            
        Returns:
            obj: An iterator that will yields parsed news feature, in the format of dict.
        """
        if not hasattr(self, "news_index_matrix"):
            self._init_news(news_file)

        keys = ["news_index_batch"]
        keys.extend([f"candidate_{attr.replace('news_', '').replace('index', 'batch')}"
                     for attr in self.news_index_matrix.keys()])
        news_dict = {k: [] for k in keys}
        cnt = 0

        for index in range(len(self.news_title_index)):
            news_dict["news_index_batch"].append(index)
            for attr, news_index in self.news_index_matrix.items():
                attr = f"candidate_{attr.replace('news_', '').replace('index', 'batch')}"
                news_dict[attr].append(news_index[index])
            cnt += 1
            if cnt >= self.batch_size:
                yield self._convert_data(news_dict)
                news_dict = {k: [] for k in keys}
                cnt = 0

        if cnt > 0:
            yield self._convert_data(news_dict)

    def load_impression_from_file(self, behaviors_file, test_set=False):
        """Read and parse impression data from behaivors file.
        
        Args:
            behaviors_file (str): A file contains several informations of behaviros.

        Returns:
            obj: An iterator that will yields parsed impression data, in the format of dict.
        """

        if not hasattr(self, "histories"):
            self._init_behaviors(behaviors_file, test_set=test_set)

        indexes = np.arange(len(self.imprs))

        for index in indexes:
            if not test_set:
                impr_label = np.array(self.labels[index], dtype="int32")
                impr_news = np.array(self.imprs[index], dtype="int32")

                yield self.impr_indexes[index], impr_news, self.uindexes[index], impr_label
            else:
                impr_news = np.array(self.imprs[index], dtype="int32")

                yield self.impr_indexes[index], impr_news, self.uindexes[index]
