import json
import logging

import numpy as np
from hyperopt import fmin, tpe
from hyperopt import hp


def find_best_threshold_squadutils_v2(is_impossible_dict, na_scores, results, best_EM=True):
    with open(".data/squad2/dev-v2.0.json") as f:
        prediction_json = json.load(f)["data"]
    from src.scripts.tools.evaluate_squad_2_0 import get_raw_scores, \
        apply_no_ans_threshold, make_eval_dict, find_all_best_thresh
    na_prob_thresh = 1.0  # default value taken from the eval script
    qid_to_has_ans = {k: not (v) for k, v in is_impossible_dict.items()}
    exact_raw, f1_raw = get_raw_scores(prediction_json, results)
    exact_thresh = apply_no_ans_threshold(exact_raw, na_scores, qid_to_has_ans,
                                          na_prob_thresh)
    f1_thresh = apply_no_ans_threshold(f1_raw, na_scores, qid_to_has_ans,
                                       na_prob_thresh)
    out_eval = make_eval_dict(exact_thresh, f1_thresh)
    find_all_best_thresh(out_eval, results, exact_raw, f1_raw, na_scores, qid_to_has_ans)
    print(out_eval)
    if best_EM:
        tau = out_eval['best_exact_thresh']
    else:
        tau = out_eval['best_f1_thresh']
    return tau


def find_best_threshold_hyperopt(impossible_d, scores, search_range=[-10, 10]):
    gt, local_scores = [], []
    for id in impossible_d.keys():
        gt.append(impossible_d[id])
        local_scores.append(scores[id])
    gt = np.array(gt)
    local_scores = np.array(local_scores)

    def objective(tau):
        return -np.mean((local_scores > tau) == gt)

    space = hp.uniform('tau', search_range[0], search_range[-1])

    loggers_to_shut_up = [
        "hyperopt.tpe",
        "hyperopt.fmin",
        "hyperopt.pyll.base",
    ]

    for logger in loggers_to_shut_up:
        logging.getLogger(logger).setLevel(logging.ERROR)

    best = fmin(objective, space, algo=tpe.suggest, max_evals=2000, verbose=False)
    return best['tau']


if __name__ == "__main__":
    with open("is_impossible_dict.json", "r") as f:
        is_impossible_d = json.load(f)
    with open(".data/squad2/null_logods.json", "r") as f:
        noansdiffs = json.load(f)

    tau = find_best_threshold_hyperopt(is_impossible_d, noansdiffs)
    print(tau)
