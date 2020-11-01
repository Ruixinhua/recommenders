import torch
import os

import yaml
import torch.optim as optim
import shutil
from datetime import datetime
import torch.backends.cudnn as cudnn
import importlib

seed = 42
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
name_mapping = {"nrms": "NRMSModel", "nrms_entity": "NRMSModelEntity"}


def load_config(filename):
    with open(filename, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def print_log(info_str, other_info='', file=None):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file:
        print("%s [INFO]: %s %s" % (current_time, info_str, other_info), file=file)
        file.flush()
    else:
        print("%s [INFO]: %s %s" % (current_time, info_str, other_info))


def copy_files(source_paths, des_paths, is_debug=False):
    """
    copy files from source to destination
    """
    for source_path, des_path in zip(source_paths, des_paths):
        if not os.path.exists(os.path.dirname(des_path)):
            os.makedirs(os.path.dirname(des_path))
        shutil.copyfile(source_path, des_path)
        if is_debug:
            print_log("Copy file from %s to %s" % (source_path, des_path))


def get_device(i=0):
    """
    setup GPU device if available, move models into configured device
    """
    if torch.cuda.is_available():
        return torch.device("cuda:%d" % i)
    else:
        return torch.device("cpu")


def get_model_class(model_type="nrms", **model_params):
    model_object = importlib.import_module(f"reco_utils.recommender.newsrec.models.{model_type}")
    model_class = getattr(model_object, name_mapping[model_type])
    return model_class(**model_params)


def get_model_by_state(state_dic_path, model_class, device=get_device()):
    if state_dic_path is not None and os.path.exists(state_dic_path):
        model_class.load_state_dict(torch.load(state_dic_path, map_location=device))
    return model_class


def get_model_opt(state_dic_path, model_class, learning_rate=1e-3, device=get_device()):
    model = get_model_by_state(state_dic_path, model_class, device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer
