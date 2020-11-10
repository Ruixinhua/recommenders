import os
import torch
# set up configuration
from configuration import (
    data_path, train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, MIND_type, load_trainer
)
from utils import tools
import argparse

parse = argparse.ArgumentParser(description="NRMS Training process")
parse.add_argument("--log" "-l", dest="log_file", metavar="TEXT", help="log file", default="log.txt")
parse.add_argument("--configure", "-c", dest="config", metavar="TEXT", help="yaml file", default=r"nrms.yaml")
parse.add_argument("--device_id", "-d", dest="device_id", metavar="INTEGER", default=0)
args = parse.parse_args()
trainer = load_trainer(args.config, log_file=args.log_file, device_id=args.device_id)
# save model
model_path = os.path.join(data_path, "checkpoint")
os.makedirs(model_path, exist_ok=True)

trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, model_dir=model_path)
trainer.model = trainer.best_model
res_syn = trainer.run_eval(valid_news_file, valid_behaviors_file)
tools.print_log(f"validation result: {res_syn}")

torch.save(trainer.model.state_dict(), os.path.join(model_path, f"nrms_{MIND_type}.pth"))
