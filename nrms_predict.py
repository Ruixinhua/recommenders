import os
import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from configuration import get_path, load_trainer, get_data_path


def write_prediction(imp_indexes, imp_preds):
    with open(f"{inference_dir}/prediction.txt", "w") as f, open(f"{inference_dir}/probability.txt", "w") as f_p:
        for impr_index, preds in tqdm(zip(imp_indexes, imp_preds)):
            impr_index += 1

            preds_list = "[" + ",".join([str(i) for i in preds]) + "]"
            f_p.write(" ".join([str(impr_index), preds_list]) + "\n")

            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = "[" + ",".join([str(i) for i in pred_rank]) + "]"
            f.write(" ".join([str(impr_index), pred_rank]) + "\n")


parse = argparse.ArgumentParser(description="Prediction process")
parse.add_argument("--configure", "-c", help="yaml file", dest="config", metavar="FILE", default=r"nrms.yaml")
parse.add_argument("--device_id", "-d", dest="device_id", metavar="INT", default=0)
parse.add_argument("--mind_type", "-t", dest="mind_type", metavar="TEXT", default="small")
parse.add_argument("--model_class", "-m", dest="model_class", metavar="TEXT", default="nrms")
args = parse.parse_args()
test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)
valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
start_time = time.time()
inference_dir = os.path.join(get_data_path(args.mind_type), "prediction")
os.makedirs(inference_dir, exist_ok=True)
# set trainer
trainer = load_trainer(args.config, device_id=int(args.device_id), model_class=args.model_class)
# load model
model_path = os.path.join(get_data_path(args.mind_type), "checkpoint", f"best_model.pth")
state = torch.load(model_path, map_location=trainer.device)
trainer.model.load_state_dict(state)
with torch.no_grad():
    trainer.model.eval()
    # tools.print_log(trainer.run_eval(valid_news_file, valid_behaviors_file))
    group_impr_indexes, group_preds = trainer.run_fast_eval(test_news_file, test_behaviors_file, test_set=True)
write_prediction(group_impr_indexes, group_preds)
print(f"inference on category: cost: {time.time() - start_time}s.")
