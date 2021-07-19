import logging
import os
import sys
import torch

from src.datasets.utility import download_url
from src.scripts.tools.evaluate_adversarial_squad import evaluate_advsquad
from src.scripts.tools.utility import setup_logging, mkdir
from src.scripts.trainers.LRMtrainer import QAFramework

filtering_type = "_filtered"
config = {
    "objective_variant": "independent_joint",
    "decoding_variant": "joint" + filtering_type,

    "model_type": "bert-base-uncased",
    "tokenizer_type": "bert-base-uncased",

    "similarity_function": "simple_multi",  # None,

    "max_answer_length": 30,
    "validation_batch_size": 16,

    "dataset": "squad_transformers",

    "tensorboard_logging": False,
    "cache_dir": ".Transformers_cache",

    "adjust_by_one": False,
    "multi_gpu": False,
    "validate_only": True,
    "model_to_validate": ".checkpoints/checkpoint_squad_transformers_bert-base-uncased_compound_EM_82.25_F1_88.77.pt.sd"
}
if __name__ == "__main__":
    setup_logging(os.path.basename(sys.argv[0]).split(".")[0],
                  logpath=".logs/",
                  config_path="configurations/logging.yml")
    if len(sys.argv) > 1:
        default_path = os.getcwd()
        project_path = os.path.abspath(os.path.join(os.path.realpath(__file__), "../../"))
        sys.path.append(project_path)
        os.chdir(project_path)
        print(f"Working directory and pypath set to {project_path}")

        config["tokenizer_type"] = config["model_type"] = sys.argv[1]

        config["objective_variant"] = sys.argv[2]
        config["decoding_variant"] = ("joint" if "joint" in sys.argv[2] else "independent") + config["decoding_variant"]

        logging.info(f"Objective variant: {config['objective_variant']}")
        logging.info(f"Decoding variant: {config['decoding_variant']}")

        CHECKPOINT_DIR = sys.argv[-1]
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
        logging.info("Found checkpoints:")
        for ch in checkpoints:
            logging.info(ch)
    else:
        checkpoints = [config["model_to_validate"]]

    ADV_S_CACHEDIR = ".data/adv_squad/"
    ADV_S_ADD_SENT = os.path.join(ADV_S_CACHEDIR, "sample1k-HCVerifyAll.json")
    ADV_S_ADD_ONE_SENT = os.path.join(ADV_S_CACHEDIR, "sample1k-HCVerifySample.json")

    ADV_S_ADD_SENT_URL = "https://worksheets.codalab.org/rest/bundles/0xb765680b60c64d088f5daccac08b3905/contents/blob/"
    ADV_S_ADD_ONESENT_URL = "https://worksheets.codalab.org/rest/bundles/0x3ac9349d16ba4e7bb9b5920e3b1af393/contents/blob/"

    PREDICTION_FILE = "advs_predictions.json"
    if not os.path.exists(ADV_S_CACHEDIR):
        mkdir(ADV_S_CACHEDIR)
        download_url(ADV_S_ADD_SENT, ADV_S_ADD_SENT_URL)
        download_url(ADV_S_ADD_ONE_SENT, ADV_S_ADD_ONESENT_URL)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device)
          + "\n - device name: " + torch.cuda.get_device_name(device=device)
          + "\n - total_memory: " + str(torch.cuda.get_device_properties(device).total_memory / 1e6) + " GB")

    prediction_path = os.path.join(ADV_S_CACHEDIR, PREDICTION_FILE)
    try:
        for checkpoint in checkpoints:
            logging.info(f"Running evaluation of {checkpoint}")
            if len(sys.argv) > 1:
                config["model_to_validate"] = os.path.join(CHECKPOINT_DIR, checkpoint)
            framework = QAFramework(config, device, force_local=True)
            framework.predict(ADV_S_ADD_SENT, prediction_path)
            result = evaluate_advsquad(ADV_S_ADD_SENT, prediction_path)
            logging.info("ADD_SENT")
            logging.info(result)

            result = evaluate_advsquad(ADV_S_ADD_ONE_SENT, prediction_path)
            logging.info("ADD_ONE_SENT")
            logging.info(result)

    except BaseException as be:
        logging.error(be)
        raise be
