import os
import json

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

epochs = 8
seed = 42


def get_root_path():
    config = json.load(open("config.json"))
    return config["data_root_path"]


def get_data_path(mind_type="small"):
    return os.path.join(get_root_path(), mind_type)


def get_path(name, mind_type="small"):
    # set up the path of dataset
    news_file = os.path.join(get_data_path(mind_type), name, r'news.tsv')
    behaviors_file = os.path.join(get_data_path(mind_type), name, r'behaviors.tsv')
    return news_file, behaviors_file


def get_emb_path():
    return os.path.join(get_root_path(), "utils", "embedding.npy")


def get_user_dic_path():
    return os.path.join(get_root_path(), "utils", "uid2index.pkl")


def get_dict_file():
    return os.path.join(get_root_path(), "utils", "word_dict.pkl")


def get_yaml_path(yaml_name=r"nrms.yaml"):
    return os.path.join(get_root_path(), "utils", yaml_name)


def download_data(mind_type="small"):
    mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)
    data_path = get_data_path(mind_type)
    train_news_file, _ = get_path("train", mind_type)
    valid_news_file, _ = get_path("valid", mind_type)
    test_news_file, _ = get_path("test", mind_type)
    if not os.path.exists(train_news_file):
        download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)
    if mind_type == "large":
        if not os.path.exists(test_news_file):
            download_deeprec_resources(mind_url, os.path.join(data_path, 'test'), mind_dev_dataset)
    if not os.path.exists(get_yaml_path()):
        utils_url = r'https://recodatasets.blob.core.windows.net/newsrec/'
        download_deeprec_resources(utils_url, os.path.join(get_root_path(), 'utils'), mind_utils)


def get_params(yaml_name, device_id=0, model_class=None):
    if yaml_name:
        yaml_path = os.path.join(get_root_path(), "utils", yaml_name)
    else:
        yaml_path = get_yaml_path()

    return prepare_hparams(yaml_path, wordEmb_file=get_emb_path(), wordDict_file=get_dict_file(), device_id=device_id,
                           epochs=epochs, show_step=10, userDict_file=get_user_dic_path(), model_class=model_class)


def load_trainer(yaml_name=None, log_file=None, device_id=0, model_class=None, mind_type="small"):
    log_path = os.path.join(get_data_path(mind_type), "log")
    download_data(mind_type)
    os.makedirs(log_path, exist_ok=True)
    hparams = get_params(yaml_name, device_id, model_class)
    # set up log file
    log_file = log_file if log_file else hparams.log_file
    log_file = os.path.join(log_path, log_file)
    hparams.log_file = log_file
    print(hparams.to_string())
    # setup different trainer
    if hparams.trainer == "entity":
        from reco_utils.recommender.newsrec.trainers.entity_trainer import EntityTrainer
        trainer = EntityTrainer(hparams, MINDIterator, seed)
    elif hparams.trainer == "body":
        from reco_utils.recommender.newsrec.trainers.body_trainer import BodyTrainer
        trainer = BodyTrainer(hparams, MINDIterator, seed)
    else:
        from reco_utils.recommender.newsrec.trainers.base_trainer import BaseTrainer
        # set trainer
        trainer = BaseTrainer(hparams, MINDIterator, seed)
    return trainer
