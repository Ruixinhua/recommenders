import os
import torch
import numpy as np
import time
from tqdm import tqdm

from configuration import (
    seed, data_path, test_news_file, test_behaviors_file, valid_news_file, valid_behaviors_file, load_trainer
)

from utils import tools


def write_prediction(imp_indexes, imp_preds):
    with open(f"{inference_dir}/prediction.txt", "w") as f, open(f"{inference_dir}/probability.txt", "w") as f_p:
        for impr_index, preds in tqdm(zip(imp_indexes, imp_preds)):
            impr_index += 1

            preds_list = "[" + ",".join([str(i) for i in preds]) + "]"
            f_p.write(" ".join([str(impr_index), preds_list]) + "\n")

            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = "[" + ",".join([str(i) for i in pred_rank]) + "]"
            f.write(" ".join([str(impr_index), pred_rank]) + "\n")


start_time = time.time()
inference_dir = f"{data_path}/prediction"
os.makedirs(inference_dir, exist_ok=True)
# set trainer
yaml_name = r"nrms.yaml"
trainer = load_trainer(yaml_name)
# load model
model_path = os.path.join(data_path, "checkpoint", f"best_model_nrms.pth")
state = torch.load(model_path)
trainer.model.load_state_dict(state)
with torch.no_grad():
    tools.print_log(trainer.run_eval(valid_news_file, valid_behaviors_file))
    # group_impr_indexes, group_preds = trainer.run_fast_eval(test_news_file, test_behaviors_file, test_set=True)
# write_prediction(group_impr_indexes, group_preds)
print(f"inference on category: cost: {time.time() - start_time}s.")
