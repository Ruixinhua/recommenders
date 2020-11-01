import os
import torch
# set up configuration
from configuration import (
    trainer, data_path, train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, MIND_type
)
from utils import tools

# save model
model_path = os.path.join(data_path, "checkpoint")
os.makedirs(model_path, exist_ok=True)

trainer.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file, model_dir=model_path)
trainer.model = trainer.best_model
res_syn = trainer.run_eval(valid_news_file, valid_behaviors_file)
tools.print_log(f"validation result: {res_syn}")

torch.save(trainer.model.state_dict(), os.path.join(model_path, f"nrms_{MIND_type}.pth"))
