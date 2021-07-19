import logging
import os
import sys
import traceback
import torch

from src.scripts.trainers.LRMtrainer import QAFramework
from src.scripts.tools.utility import setup_logging

config = {
    "objective_variant": "joint",  # "independent", "conditional", "joint", "independent_joint"
    "decoding_variant": "joint_filtered",
    # [independent|joint] (no filtering), [independent|joint]_filtered (length filtering), [independent|joint]_filtered_combined (length + SFF filtering)

    "similarity_function": "simple_multi",  # simple_multi, ff_mlp, multi, add, multi_add

    # "tokenizer_type": "albert-xxlarge-v1",
    # "model_type": "albert-xxlarge-v1",

    "tokenizer_type": "bert-base-uncased",
    "model_type": "bert-base-uncased",

    # hyperparams for BERT from the paper
    "epochs": 2,
    "batch_size": 6,
    "validation_batch_size": 6,
    "true_batch_size": 12,
    "max_grad_norm": 1.0,
    "weight_decay": 0.0,
    "learning_rate": 3e-05,
    "adam_eps": 1e-08,
    "warmup_steps": 0,

    # hyperparams for BERT-large from the paper
    # "epochs": 2,
    # "batch_size": 1,
    # "true_batch_size": 48,
    # "max_grad_norm": 1.0,
    # "weight_decay": 0.0,
    # "learning_rate": 5e-05,
    # "adam_eps": 1e-08,
    # "warmup_steps": 0,

    # hyperparams for ALBERT-xxlarge-v1 from the paper
    # "epochs": 2,
    # "batch_size": 1,
    # "true_batch_size": 48,
    # "max_grad_norm": 1.0,
    # "weight_decay": 0.01,
    # "learning_rate": 5e-05,
    # "adam_eps": 1e-06,
    # "warmup_proportion": 0.1,

    "max_answer_length": 30,

    "dataset": "squad_transformers",  # "squad_transformers","squad2_transformers"

    "tensorboard_logging": False,
    "cache_dir": ".Transformers_cache",

    "save_only_last_epoch": True,
    "save_dir": ".saved",
    "multi_gpu": False,
    "validate_only": False,

    "model_to_validate": ""
}

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    device = torch.device(
        f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        framework = QAFramework(config, device)
        framework.fit()
    except BaseException as be:
        logging.error(be)
        logging.error(traceback.format_exc())
        raise be
