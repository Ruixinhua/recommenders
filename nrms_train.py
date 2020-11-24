import os
# set up configuration
from configuration import get_path, load_trainer, get_data_path
from utils import tools
import argparse

parse = argparse.ArgumentParser(description="NRMS Training process")
parse.add_argument("--log", "-l", dest="log_file", metavar="FILE", help="log file", default="log.txt")
parse.add_argument("--configure", "-c", dest="config", metavar="FILE", help="yaml file", default=r"nrms.yaml")
parse.add_argument("--device_id", "-d", dest="device_id", metavar="INT", default=0)
parse.add_argument("--model_class", "-m", dest="model_class", metavar="TEXT", default="nrms")
parse.add_argument("--mind_type", "-t", dest="mind_type", metavar="TEXT", default="small")
parse.add_argument("--resume", "-r", dest="resume", metavar="INT", help="whether resume or not", default=0)
args = parse.parse_args()

train_news_file, train_behaviors_file = get_path("train", mind_type=args.mind_type)
valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
config = {"yaml_name": args.config, "log_file": args.log_file, "device_id": int(args.device_id),
          "model_class": args.model_class}
trainer = load_trainer(**config)
# save model
model_path = os.path.join(get_data_path(mind_type=args.mind_type), "checkpoint")
os.makedirs(model_path, exist_ok=True)
resume = int(args.resume)
resume_path = os.path.join(model_path, "best_model.pth")
if resume:
    trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, model_dir=model_path,
                resume_path=resume_path)
else:
    trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, model_dir=model_path)

trainer.model = trainer.best_model
res_syn = trainer.run_eval(valid_news_file, valid_behaviors_file)
tools.print_log(f"validation result: {res_syn}")
