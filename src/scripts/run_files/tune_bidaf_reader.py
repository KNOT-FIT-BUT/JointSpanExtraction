import copy
import logging
import math
import pickle
import traceback
import torch
import os
import sys
import numpy as np
from random import randint
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK, STATUS_FAIL
from src.scripts.tools.utility import setup_logging, set_seed, mkdir
from src.scripts.trainers.bidaf_trainer import BidafModelFramework

bidaf_config = {"objective_variant": "independent_joint",
                "batch_size": 40,
                "true_batch_size": 40,
                "decoding_variant": "joint_filtered",
                "similarity_function": "multi_add",  # multi, add, simple_multi, multi_add, mlp
                "max_answer_length": 30,

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
                "dropout_rate": 0.2,
                "RNN_nhidden": 100,
                "RNN_layers": 1,
                "char_embeddings": True,

                # Embedding options
                "optimize_embeddings": False,
                "scale_emb_grad_by_freq": False,

                # Optimization options
                "learning_rate": 0.003,
                "optimizer": "adam",
                "patience": 5,

                # Exponential moving average, as in original paper
                "ema": False,
                "start_ema_from_it": 0,
                "exp_decay_rate": 0.999,

                "validate_only": False,
                }


class FitWrapper:
    def __init__(self, framework, default_config, **cached_inputs):
        self.fw = framework
        self.default_config = default_config
        self.cached_inputs = cached_inputs


def run_hyperparam_opt(**cached_inputs):
    def run_trials():
        trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
        max_trials = 5  # initial max_trials. put something small to not have to wait
        HP_DIR = ".hp_optimization"
        TRIALS_FILE = "bidaf_comopound_hyperparaam_trials.hyperopt"
        mkdir(HP_DIR)
        TRIALS_PATH = os.path.join(HP_DIR, TRIALS_FILE)

        try:  # try to load an already saved trials object, and increase the max
            trials = pickle.load(open(TRIALS_PATH, "rb"))
            logging.info("#" * 30)
            logging.info("#" * 30)
            logging.info("Found saved Trials! Loading...")
            max_trials = len(trials.trials) + trials_step
            logging.info(
                "Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
        except:  # create a new trials object and start searching
            trials = Trials()
            logging.info("Previous trials not found! Creating new trials...")

        wrapper = FitWrapper(framework=BidafModelFramework, default_config=bidaf_config, **cached_inputs)
        model_space = {"dropout_rate": hp.uniform("cls_dropout", low=0.0, high=0.4),
                       "learning_rate": hp.loguniform('learning_rate',
                                                      low=np.log(1e-4),
                                                      high=np.log(1e-2)),
                       "true_batch_size": hp.choice("true_batch_size", [40, 80, 120]),
                       "scheduler": hp.choice("scheduler", [None, "linear"]),
                       "warmup_proportion": hp.choice("warmup_proportion", [0.0, 0.05, 0.1, 0.2]),
                       "epochs": 3 + hp.randint("epochs", 30),
                       }

        best = fmin(fn=lambda a: obj(wrapper, a), space=model_space, algo=tpe.suggest, max_evals=max_trials,
                    trials=trials)

        logging.info("Best:\n" + str(best))

        # save the trials object
        with open(TRIALS_PATH, "wb") as f:
            pickle.dump(trials, f)

    # loop indefinitely and stop whenever you like
    while True:
        run_trials()


def obj(self, opt_config):
    try:
        seed = randint(0, 10_000)
        set_seed(seed)
        logging.info(f"Random seed: {seed}")

        logging.info(f"Opt config:\n{opt_config}")
        config = copy.deepcopy(self.default_config)
        config.update(opt_config)
        config["batch_size"] = config["true_batch_size"]
        config["epochs"] = int(config["epochs"])

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        framework = self.fw(config, device)
        EM, best_model_p = framework.fit(**self.cached_inputs)
        stats = {
            "status": STATUS_OK,
            'loss': -EM,
            'accuracy': EM,
            'model_path': best_model_p
        }

    except BaseException as be:
        logging.error(be)
        logging.error(traceback.format_exc())
        return {'loss': math.inf, 'status': STATUS_FAIL}

    return stats


if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    logging.info("Precaching input data...")
    train, val, fields = BidafModelFramework.load_data(bidaf_config)
    run_hyperparam_opt(train=train, val=val, fields=fields)
