import os

from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources
from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.newsrec.newsrec_utils import get_mind_data_set

epochs = 8
seed = 42
MIND_type = 'demo'
data_root_path = 'C:\\Users\\Rui\\Documents\\Phd_research_RS\\baseline\\recommenders\\mind_dataset'
data_path = f'{data_root_path}\\{MIND_type}'

# set up the path of dataset
train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
test_news_file = os.path.join(data_path, 'test', r'news.tsv')
test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')
# TODO: modify the embedding file
wordEmb_file = os.path.join(data_root_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_root_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_root_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_root_path, "utils", r'nrms.yaml')

mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(MIND_type)

if not os.path.exists(train_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)

if not os.path.exists(valid_news_file):
    download_deeprec_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)

if not os.path.exists(yaml_file):
    utils_url = r'https://recodatasets.blob.core.windows.net/newsrec/'
    download_deeprec_resources(utils_url, os.path.join(data_root_path, 'utils'), mind_utils)

log_path = os.path.join(data_path, "log")
os.makedirs(log_path, exist_ok=True)
log_file = os.path.join(log_path, "log.txt")
# pick attribute from title, entity, vert, subvert, abstract, and define their size
news_attr = {"title": 30, "entity": 30}
model_type = "nrms_entity"
hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, wordDict_file=wordDict_file, model_type=model_type,
                          epochs=epochs, show_step=10, userDict_file=userDict_file, log_file=log_file,
                          news_attr=news_attr)
print(hparams.to_string())
if hparams.model_type == "nrms":
    from reco_utils.recommender.newsrec.trainers.base_trainer import BaseTrainer
    # set trainer
    trainer = BaseTrainer(hparams, MINDIterator, seed)
elif hparams.model_type == "nrms_entity":
    from reco_utils.recommender.newsrec.trainers.entity_trainer import EntityTrainer
    trainer = EntityTrainer(hparams, MINDIterator, seed)
