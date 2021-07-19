import logging
import traceback
from random import randint

import torch
import os
import sys

from src.scripts.tools.utility import setup_logging, set_seed
from src.scripts.trainers.bidaf_trainer import BidafModelFramework

bidaf_config = {"objective_variant": "joint",  # independent joint independent_joint
                "batch_size": 1,
                "decoding_variant": "joint_filtered",
                # [independent|joint] (no filtering), [independent|joint]_filtered (length filtering), [independent|joint]_filtered_combined (length + SFF filtering)
                "similarity_function": "ff_mlp",  # multi, add, simple_multi, multi_add, ff_mlp
                "max_answer_length": 30,

                "save_dir": ".saved",
                "result_dir": "results",
                "min_em_to_save": 65,

                "mask_rnns": True,
                "use_start_end_char_boundaries": False,
                # "validate_only": False,

                # Model architecture options
                "embedding_size": 100,
                "char_embedding_size": 16,
                "highway_layers": 2,
                "highway_dim1": 100,
                "highway_dim2": 100,
                "char_channel_size": 100,
                "char_channel_width": 5,
                "char_maxsize_vocab": 260,  # as in official AllenNLP impl
                "RNN_input_dim": 100,
                "RNN_nhidden": 100,
                "RNN_layers": 1,
                "char_embeddings": True,

                # Embedding options
                "optimize_embeddings": False,
                "scale_emb_grad_by_freq": False,

                # Optimization options
                "max_iterations": 10000,
                "optimizer": "adam",
                "patience": 5,

                # Exponential moving average, as in original paper
                "ema": False,
                "start_ema_from_it": 0,
                "exp_decay_rate": 0.999,

                "validate_only": False,
                "model_path": "bidaf_66.31_75.41.pt",

                # Parameters tuned via hyperopt
                'dropout_rate': 0.22239950646838239,
                'epochs': 21,
                'learning_rate': 0.0017592470208501842,
                'scheduler': 'linear',
                'true_batch_size': 120,
                'warmup_proportion': 0.2
                }

if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")

    seed = randint(0, 10_000)
    set_seed(seed)
    logging.info(f"Random seed: {seed}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        framework = BidafModelFramework(bidaf_config, device)
        framework.fit()
    except BaseException as be:
        logging.error(be)
        logging.error(traceback.format_exc())
