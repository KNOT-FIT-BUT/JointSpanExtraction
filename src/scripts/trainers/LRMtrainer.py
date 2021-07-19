import csv
import json
import logging
import math
import socket
import sys
import time
import torch
import torch.nn.functional as F
import transformers
from collections import defaultdict, OrderedDict
from typing import Union
from torch.nn import DataParallel
from torchtext.data import BucketIterator, Iterator
from tqdm import tqdm
from transformers import AlbertTokenizer, BertForQuestionAnswering, BertTokenizer, PreTrainedModel, \
    get_linear_schedule_with_warmup, AlbertForQuestionAnswering, BertModel, AlbertModel
from src.datasets.utility import VALIDATION_F_MRQA_SEARCHQA, TRAIN_F_MRQA_SEARCHQA, VALIDATION_F_MRQA_TRIVIAQA, \
    TRAIN_F_MRQA_TRIVIAQA, VALIDATION_F_MRQA_NEWSQA, TRAIN_F_MRQA_NEWSQA, VALIDATION_F_MRQA_NQ, TRAIN_F_MRQA_NQ, \
    VALIDATION_F_MRQA_SQUAD, TRAIN_F_MRQA_SQUAD, VALIDATION_F_SQUAD, TRAIN_F_SQUAD, TRAIN_F_SQUAD2, VALIDATION_F_SQUAD2
from src.datasets.SQUAD2_dataset_transformers import Squad2Dataset_transformers
from src.datasets.SQUAD_dataset_transformers import SquadDataset_transformers
from src.datasets.MRQA_dataset_transformers import MRQADataset_transformers

from src.datasets.utility import VALIDATION_F_SQUAD, TRAIN_F_SQUAD, TRAIN_F_SQUAD2, VALIDATION_F_SQUAD2
from src.models.LRM.transformer_indep_joint_for_QA import TransformerForQuestionAnsweringIndepJoint
from src.models.LRM.transformer_joint_for_QA import TransformerForQuestionAnsweringJoint
from src.models.LRM.transformer_conditional_for_QA import TransformerForQuestionAnsweringConditional

from src.models.utils.modelparts import SpanPredictionModule
from src.scripts.tools.compute_predictions_from_logits import get_predictions
from src.scripts.tools.evaluate_squad_1_1 import evaluate_squad_11
from src.scripts.tools.evaluate_squad_2_0 import evaluate_squad_20
from src.scripts.tools.evaluate_mrqa import evaluate_mrqa

from src.scripts.tools.utility import count_parameters, report_parameters, get_timestamp, mkdir


class QAFramework:
    def __init__(self, config, device, force_local=False):
        mkdir("results")
        mkdir(".saved")
        self.config = config
        self.device = device
        self.n_iter = 0
        if config["tokenizer_type"].startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(config["tokenizer_type"], cache_dir=self.config["cache_dir"],
                                                           local_files_only=force_local)
        elif config["tokenizer_type"].startswith("albert"):
            self.tokenizer = AlbertTokenizer.from_pretrained(config["tokenizer_type"],
                                                             cache_dir=self.config["cache_dir"],
                                                             local_files_only=force_local)
        if config["tensorboard_logging"]:
            from torch.utils.tensorboard import SummaryWriter
            self.boardwriter = SummaryWriter()

    def train_epoch(self, model: PreTrainedModel, optimizer: torch.optim.Optimizer,
                    scheduler: Union[torch.optim.lr_scheduler.LambdaLR, None], train_iter: Iterator) -> float:
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        updated = False
        it = tqdm(train_iter, total=len(train_iter.data()) // train_iter.batch_size + 1)
        model.mask_cls = False  # never mask cls during training
        for i, batch in enumerate(it):
            updated = False
            if self.config["objective_variant"] == "independent":
                outputs = model(batch.input,
                                token_type_ids=batch.segment_ids,
                                attention_mask=batch.input_padding)
                logprobs_S, logprobs_E = outputs

                logprobs_S, logprobs_E = self.adjust_for_independent(batch, logprobs_S, logprobs_E,
                                                                     train_iter)

                lossf = lambda logits, targets: F.cross_entropy(logits, targets, reduction="mean") \
                    if self.config["dataset"] in ["squad_transformers", "mrqa_transformers",
                                                  "squad2_transformers"] else \
                    lambda logits, targets: QAFramework.masked_cross_entropy(logits, targets, reduce=True)

                loss_s, loss_e = lossf(logprobs_S, batch.a_start), \
                                 lossf(logprobs_E, batch.a_end)

                loss = (loss_s + loss_e) / 2.
            elif self.config["objective_variant"] == "joint":
                logprobs_J = model(batch.input,
                                   token_type_ids=batch.segment_ids,
                                   attention_mask=batch.input_padding)

                loss = QAFramework.masked_cross_entropy(logprobs_J.view(logprobs_J.shape[0], -1),
                                                        batch.a_start * batch.input.shape[-1] + batch.a_end)

            elif self.config["objective_variant"] == "independent_joint":
                outputs = model(batch.input,
                                token_type_ids=batch.segment_ids,
                                attention_mask=batch.input_padding)
                logprobs_S, logprobs_E, logprobs_J = outputs

                logprobs_S, logprobs_E = self.adjust_for_independent(batch, logprobs_S, logprobs_E,
                                                                     train_iter)
                lossf = lambda logits, targets: F.cross_entropy(logits, targets, reduction="mean") \
                    if self.config["dataset"] in ["squad_ow", "mrqa_ow", "squad2_ow",
                                                  "squad_transformers", "mrqa_transformers",
                                                  "squad2_transformers"] else \
                    lambda logits, targets: QAFramework.masked_cross_entropy(logits, targets, reduce=True)
                loss_s, loss_e = lossf(logprobs_S, batch.a_start), \
                                 lossf(logprobs_E, batch.a_end)

                loss_joint = lossf(logprobs_J.view(logprobs_J.shape[0], -1),
                                   batch.a_start * logprobs_E.shape[-1] + batch.a_end)

                loss = (loss_s + loss_e + loss_joint) / 3.
            elif self.config["objective_variant"] == "conditional":
                outputs = model(batch.input,
                                token_type_ids=batch.segment_ids,
                                attention_mask=batch.input_padding,
                                a_starts=batch.a_start)
                logprobs_S, logprobs_E = outputs

                logprobs_S, logprobs_E = self.adjust_for_independent(batch, logprobs_S, logprobs_E,
                                                                     train_iter)

                lossf = lambda logits, targets: F.cross_entropy(logits, targets, reduction="mean") \
                    if self.config["dataset"] in ["squad_ow", "mrqa_ow", "squad2_ow",
                                                  "squad_transformers", "mrqa_transformers",
                                                  "squad2_transformers"] else \
                    lambda logits, targets: QAFramework.masked_cross_entropy(logits, targets, reduce=True)

                loss_s, loss_e = lossf(logprobs_S, batch.a_start), \
                                 lossf(logprobs_E, batch.a_end)

                loss = (loss_s + loss_e) / 2.
            loss = loss / update_ratio
            loss.backward()
            if (i + 1) % update_ratio == 0:
                self.n_iter += 1
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                               self.config["max_grad_norm"])
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                updated = True
                tr_loss = train_loss / (i + 1)
                it.set_description(f"Training loss: {tr_loss}")

                if self.config["tensorboard_logging"] and self.n_iter % 100 == 0:  # Log tboard after every 100 updates
                    self.boardwriter.add_scalar('Online_Loss/train', tr_loss, global_step=self.n_iter)
                    self.boardwriter.add_scalar('Stochastic_Loss/train', loss.item(), global_step=self.n_iter)
                    # grad norms w.r.expectation_embeddings_unilm input dimensions
                    for name, param in model.named_parameters():
                        if param.grad is not None and not param.grad.data.is_sparse:
                            self.boardwriter.add_histogram(f"gradients_wrt_hidden_{name}/",
                                                           param.grad.data.norm(p=2, dim=0),
                                                           global_step=self.n_iter)

            train_loss += loss.item()

        # Do the last step if needed with what has been accumulated
        if not updated:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        return train_loss / len(train_iter.data())

    @torch.no_grad()
    def predict(self, inpf, outf):
        if self.config["dataset"] == "squad_transformers":
            dataset = SquadDataset_transformers
        elif self.config["dataset"] == "squad2_transformers":
            dataset = Squad2Dataset_transformers
        elif self.config["dataset"] == "mrqa_transformers":
            dataset = MRQADataset_transformers
        else:
            raise NotImplementedError(f"Unknown dataset {self.config['dataset']}")

        fields = dataset.prepare_fields(pad_t=self.tokenizer.pad_token_id)

        test = dataset(inpf, self.tokenizer, fields, transformer=self.config["model_type"], cachedir=".",
                       is_training=False)

        test_iter = BucketIterator(test, sort_key=lambda x: (len(x.input)), sort=True,
                                   batch_size=self.config["validation_batch_size"],
                                   repeat=False,
                                   device=self.device)
        model = self.load_model(self.config["model_to_validate"])
        model.eval()

        hacks = {
            "max_answer_length": self.config["max_answer_length"]
            if "filtered" in self.config['decoding_variant'] else 9e9
        }

        model.mask_cls = self.config["dataset"] != "squad2_transformers"
        all_results = []
        n_best_size = 120
        for i, batch in tqdm(enumerate(test_iter), total=len(test_iter.data()) // test_iter.batch_size + 1):
            if "combined" in self.config['decoding_variant']:
                # get doc input for probability recombination via surface form filtering
                doc_inputs = QAFramework.get_doc_inp(batch)
                hacks["combine_surface_forms"] = (True, doc_inputs)
            if self.config["objective_variant"] == "independent":
                logprobs_S, logprobs_E = model(batch.input,
                                               token_type_ids=batch.segment_ids,
                                               attention_mask=batch.input_padding)
                # Masking is not done within vanilla model
                mask = ~batch.segment_ids.bool()
                if not model.mask_cls:
                    mask[:, 0] = False
                logprobs_S.masked_fill_(mask, -math.inf)
                logprobs_E.masked_fill_(mask, -math.inf)

                best_span_scores, (start_indexes, end_indexes) = SpanPredictionModule.decode_topN_with_hacks(
                    logprobs_S,
                    logprobs_E,
                    score="logprobs",
                    N=n_best_size,
                    has_sentinel=True, hacks=hacks)

                feature_null_score = logprobs_S[:, 0] + logprobs_E[:, 0]
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         best_span_scores[r_idx].cpu().tolist(),
                         start_indexes[r_idx].cpu().tolist(), end_indexes[r_idx].cpu().tolist())
                    )

            elif self.config["objective_variant"] == "joint":
                logprobs_J = model(batch.input,
                                   token_type_ids=batch.segment_ids,
                                   attention_mask=batch.input_padding)

                best_span_scores, (start_indexes, end_indexes) = SpanPredictionModule.decode_topN_joint_wth_hacks(
                    logprobs_J,
                    score="logprobs",
                    N=n_best_size,
                    has_sentinel=True, hacks=hacks)

                feature_null_score = logprobs_J[:, 0, 0]
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         best_span_scores[r_idx].cpu().tolist(),
                         start_indexes[r_idx].cpu().tolist(), end_indexes[r_idx].cpu().tolist())
                    )
            elif self.config["objective_variant"] == "independent_joint":
                logprobs_S, logprobs_E, logprobs_J = model(batch.input,
                                                           token_type_ids=batch.segment_ids,
                                                           attention_mask=batch.input_padding)

                best_span_scores, (start_indexes, end_indexes) = SpanPredictionModule.decode_topN_joint_wth_hacks(
                    logprobs_J,
                    score="logprobs",
                    N=n_best_size,
                    has_sentinel=True, hacks=hacks)

                feature_null_score = logprobs_J[:, 0, 0]
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         best_span_scores[r_idx].cpu().tolist(),
                         start_indexes[r_idx].cpu().tolist(), end_indexes[r_idx].cpu().tolist())
                    )
            elif self.config["objective_variant"] == "conditional":

                start_logits, end_logits, topk_start_positions = model(
                    batch.input,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_padding)

                logprobs_S, logprobs_E0, logprobs_Emax, best_scores_f, start_indexes_f, end_indexes_f = \
                    SpanPredictionModule.decode_conditional(start_logits,
                                                            end_logits,
                                                            topk_start_positions,
                                                            model.beam_search_topn,
                                                            max_answer_length=self.config["max_answer_length"]
                                                            if "filtered" in
                                                               self.config["decoding_variant"] else 1000)

                feature_null_score = logprobs_S[:, 0].cpu()  # + logprobs_E[:, 0].cpu()
                feature_null_score_end = logprobs_E0[:, 0].cpu()  # + logprobs_E[:, 0].cpu()
                feature_null_score_end_maxn = logprobs_Emax[:, 0].cpu()  # + logprobs_E[:, 0].cpu()

                if self.config.get("experiment_disable_noans", False):
                    feature_null_score += -math.inf
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         feature_null_score_end[r_idx].cpu().tolist(),
                         feature_null_score_end_maxn[r_idx].cpu().tolist(),
                         best_scores_f[r_idx],  # .cpu().tolist(),
                         start_indexes_f[r_idx], end_indexes_f[r_idx])

                    )

        predictions = get_predictions(
            all_examples=test_iter.dataset.problem_examples,
            all_features=test_iter.dataset.problem_features,
            all_results=all_results,
            n_best_size=n_best_size,
            do_lower_case=True,
            output_prediction_file=None,
            output_nbest_file=None,
            output_null_log_odds_file=None,
            verbose_logging=False,
            version_2_with_negative=self.config["dataset"] == "squad2_transformers",
            tokenizer=self.tokenizer,
        )
        del all_results
        with open(outf, "w") as f:
            json.dump(predictions, f)

    def adjust_for_independent(self, batch, logprobs_S, logprobs_E, train_iter):
        if self.config["dataset"] in ["squad_transformers", "mrqa_transformers", "squad2_transformers"]:
            # first segment is question, in pytorch transformers
            indep_mask = (~batch.segment_ids.bool()) | \
                         (batch.input == self.tokenizer.sep_token_id) | \
                         (batch.input == self.tokenizer.pad_token_id)
            if not train_iter.dataset.skip_no_ans_w or self.config["dataset"] == "squad2_transformers":
                indep_mask[:, 0] = False  # do not mask CLS token -level output
        else:
            raise ValueError(f"Unknown dataset {self.config['dataset']}")

        logprobs_S = logprobs_S.masked_fill(indep_mask, -math.inf)
        logprobs_E = logprobs_E.masked_fill(indep_mask, -math.inf)
        return logprobs_S, logprobs_E

    @staticmethod
    def masked_cross_entropy(input, unclamped_target, reduce=True, mask_infs=True):
        """ Deals with ignored indices and inf losses """
        # Seeing inf is not an error,
        # if input is truncated and answer points to place after truncated position, that happens to be question or padding,
        # this output is masked to inf

        # Indices longer than input are clamped to ignore index and ignored
        ignored_index = input.size(1)
        target = unclamped_target.clamp(0, ignored_index)

        losses = F.cross_entropy(input, target, ignore_index=ignored_index, reduction="none")
        # get rid of inf losses
        if mask_infs:
            losses = losses.masked_fill(losses == math.inf, 0.)
        if reduce:
            return torch.sum(losses) / (torch.nonzero(losses.data).size(0) + 1e-15)
        else:
            return losses

    @torch.no_grad()
    def validate(self, model: PreTrainedModel, iter: Iterator):
        model.eval()
        lossvalues = []
        all_results = []
        n_best_size = 100

        hacks = {
            "max_answer_length": self.config["max_answer_length"]
            if "filtered" in self.config['decoding_variant'] else 9e9
        }

        if self.config["dataset"].startswith("squad2"):
            pred_folder = "squad2"
        elif self.config["dataset"].startswith("squad"):
            pred_folder = "squad"
        elif self.config["dataset"].startswith("mrqa"):
            subdataset = self.config["subdataset"]
            pred_folder = "mrqa_{}".format(subdataset)
        else:
            raise NotImplementedError(f"Unsupported dataset {self.config['dataset']}")

        model.mask_cls = not self.config["dataset"] == "squad2_transformers"

        for i, batch in tqdm(enumerate(iter), total=len(iter.data()) // iter.batch_size + 1):
            if "combined" in self.config['decoding_variant']:
                # get doc input for probability recombination via surface form filtering
                doc_inputs = QAFramework.get_doc_inp(batch)
                hacks["combine_surface_forms"] = (True, doc_inputs)

            if self.config["objective_variant"] == "independent":
                logprobs_S, logprobs_E = model(batch.input,
                                               token_type_ids=batch.segment_ids,
                                               attention_mask=batch.input_padding)
                # Masking is not done within vanilla model
                mask = ~batch.segment_ids.bool()
                if not model.mask_cls:
                    mask[:, 0] = False
                logprobs_S.masked_fill_(mask, -math.inf)
                logprobs_E.masked_fill_(mask, -math.inf)

                loss_s, loss_e = QAFramework.masked_cross_entropy(logprobs_S, batch.a_start, reduce=False), \
                                 QAFramework.masked_cross_entropy(logprobs_E, batch.a_end, reduce=False)

                loss = (loss_s + loss_e) / 2.

                best_span_scores, (start_indexes, end_indexes) = SpanPredictionModule.decode_topN_with_hacks(
                    logprobs_S,
                    logprobs_E,
                    score="logprobs",
                    N=n_best_size,
                    has_sentinel=True, hacks=hacks)

                feature_null_score = logprobs_S[:, 0] + logprobs_E[:, 0]
                if self.config.get("experiment_disable_noans", False):
                    feature_null_score += -math.inf
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         best_span_scores[r_idx].cpu().tolist(),
                         start_indexes[r_idx].cpu().tolist(), end_indexes[r_idx].cpu().tolist())
                    )

            elif self.config["objective_variant"] == "joint":
                logprobs_J = model(batch.input,
                                   token_type_ids=batch.segment_ids,
                                   attention_mask=batch.input_padding)

                loss_joint = QAFramework.masked_cross_entropy(logprobs_J.view(logprobs_J.shape[0], -1),
                                                              batch.a_start * logprobs_J.shape[-1] + batch.a_end,
                                                              reduce=False)
                loss = loss_joint

                best_span_scores, (start_indexes, end_indexes) = SpanPredictionModule.decode_topN_joint_wth_hacks(
                    logprobs_J,
                    score="logprobs",
                    N=n_best_size,
                    has_sentinel=True, hacks=hacks)

                feature_null_score = logprobs_J[:, 0, 0]
                if self.config.get("experiment_disable_noans", False):
                    feature_null_score += -math.inf
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         best_span_scores[r_idx].cpu().tolist(),
                         start_indexes[r_idx].cpu().tolist(), end_indexes[r_idx].cpu().tolist())
                    )
            elif self.config["objective_variant"] == "independent_joint":
                logprobs_S, logprobs_E, logprobs_J = model(batch.input,
                                                           token_type_ids=batch.segment_ids,
                                                           attention_mask=batch.input_padding)
                loss_s, loss_e = QAFramework.masked_cross_entropy(logprobs_S, batch.a_start, reduce=False), \
                                 QAFramework.masked_cross_entropy(logprobs_E, batch.a_end, reduce=False)

                loss_joint = QAFramework.masked_cross_entropy(logprobs_J.view(logprobs_J.shape[0], -1),
                                                              batch.a_start * logprobs_E.shape[-1] + batch.a_end,
                                                              reduce=False)
                loss = (loss_s + loss_e + loss_joint) / 3.

                best_span_scores, (start_indexes, end_indexes) = SpanPredictionModule.decode_topN_joint_wth_hacks(
                    logprobs_J,
                    score="logprobs",
                    N=n_best_size,
                    has_sentinel=True, hacks=hacks)

                feature_null_score = logprobs_J[:, 0, 0]
                if self.config.get("experiment_disable_noans", False):
                    feature_null_score += -math.inf
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         best_span_scores[r_idx].cpu().tolist(),
                         start_indexes[r_idx].cpu().tolist(), end_indexes[r_idx].cpu().tolist())
                    )
            elif self.config["objective_variant"] == "conditional":
                start_logits, end_logits, topk_start_positions = model(
                    batch.input,
                    token_type_ids=batch.segment_ids,
                    attention_mask=batch.input_padding)

                logprobs_S, logprobs_E0, logprobs_Emax, best_scores_f, start_indexes_f, end_indexes_f = \
                    SpanPredictionModule.decode_conditional(start_logits,
                                                            end_logits,
                                                            topk_start_positions,
                                                            model.beam_search_topn,
                                                            max_answer_length=self.config["max_answer_length"]
                                                            if "filtered" in
                                                               self.config["decoding_variant"] else 1000)

                feature_null_score = logprobs_S[:, 0].cpu()  # + logprobs_E[:, 0].cpu()
                feature_null_score_end = logprobs_E0[:, 0].cpu()  # + logprobs_E[:, 0].cpu()
                feature_null_score_end_maxn = logprobs_Emax[:, 0].cpu()  # + logprobs_E[:, 0].cpu()

                if self.config.get("experiment_disable_noans", False):
                    feature_null_score += -math.inf
                loss = torch.zeros(batch.batch_size)
                for r_idx in range(batch.batch_size):
                    all_results.append(
                        (batch.id[r_idx], feature_null_score[r_idx].cpu().tolist(),
                         feature_null_score_end[r_idx].cpu().tolist(),
                         feature_null_score_end_maxn[r_idx].cpu().tolist(),
                         best_scores_f[r_idx],  # .cpu().tolist(),
                         start_indexes_f[r_idx], end_indexes_f[r_idx])
                    )

            lossvalues += loss.tolist()
        prediction_file = f".data/{pred_folder}/dev_results_{socket.gethostname()}.json"
        prediction_file_nbest = f".data/{pred_folder}/dev_results_nbest_{socket.gethostname()}.json"
        predictions = get_predictions(
            all_examples=iter.dataset.problem_examples,
            all_features=iter.dataset.problem_features,
            all_results=all_results,
            n_best_size=n_best_size,
            do_lower_case=True,
            output_prediction_file=prediction_file,
            output_nbest_file=prediction_file_nbest,
            output_null_log_odds_file=f".data/{pred_folder}/null_logods.json",
            verbose_logging=False,
            version_2_with_negative=self.config["dataset"] == "squad2_transformers",
            tokenizer=self.tokenizer,
            threshold=0.0 if self.config.get("experiment_disable_noans", False) else None
        )
        del all_results

        loss = sum(lossvalues) / len(lossvalues)

        if self.config["dataset"] == "squad_transformers" or (
                self.config["dataset"] == "mrqa_transformers" and self.config["subdataset"] == "squad"):
            dataset_file = ".data/squad/dev-v1.1.json"
            expected_version = '1.1'
            with open(dataset_file) as dataset_file:
                dataset_json = json.load(dataset_file)
                if (dataset_json['version'] != expected_version):
                    logging.info('Evaluation expects v-' + expected_version +
                                 ', but got dataset with v-' + dataset_json['version'],
                                 file=sys.stderr)
                dataset = dataset_json['data']
            result = evaluate_squad_11(dataset, predictions)
            logging.info(json.dumps(result))
            em, f1 = result["exact_match"], result["f1"]
        elif self.config["dataset"] == "squad2_transformers":
            prediction_file = f".data/squad2/dev_results_{socket.gethostname()}.json"
            with open(prediction_file, "w") as f:
                json.dump(predictions, f)

            dataset_file = ".data/squad2/dev-v2.0.json"
            result = evaluate_squad_20(dataset_file, prediction_file)

            logging.info(json.dumps(result))
            em, f1 = result["exact"], result["f1"]
            hits = 0
            no_ans_gt = {example.qas_id: example.is_impossible for example in iter.dataset.problem_examples}
            for _id, a in predictions.items():
                if (a == "" and no_ans_gt[_id]) or (a != "" and not no_ans_gt[_id]):
                    hits += 1
            no_answer_accuracy = hits / len(predictions.keys())
            logging.info(f"No answer accuracy: {no_answer_accuracy}")
        elif self.config["dataset"] == "mrqa_transformers":
            if self.config["subdataset"] == "nq":
                dataset_file = ".data/mrqa/NaturalQuestionsShort.dev.jsonl"
            elif self.config["subdataset"] == "newsqa":
                dataset_file = ".data/mrqa/NewsQA.dev.jsonl"
            elif self.config["subdataset"] == "triviaqa":
                dataset_file = ".data/mrqa/TriviaQA-web.dev.jsonl"
            elif self.config["subdataset"] == "searchqa":
                dataset_file = ".data/mrqa/SearchQA.dev.jsonl"

            result = evaluate_mrqa(dataset_file, predictions)
            logging.info(json.dumps(result))
            em, f1 = result["exact_match"], result["f1"]

        else:
            raise NotImplementedError(f"Not implemented dataset: {self.config['dataset']}")
        return loss, em, f1

    @staticmethod
    def get_doc_inp(batch):
        doc_inputs = []
        for i in range(batch.input.shape[0]):
            doc_input = []
            for j in range(batch.input.shape[1]):
                if j == 0:  # Skip CLS level token
                    continue
                if batch.segment_ids[i, j] == 0:
                    doc_input.append(batch.input[i, j].item())
            doc_inputs.append(doc_input)
        return doc_inputs

    def fit(self):
        logging.debug(json.dumps(self.config, indent=4, sort_keys=True))
        if self.config["dataset"] == "squad_transformers":
            dataset = SquadDataset_transformers
            in_train = TRAIN_F_SQUAD
            in_dev = VALIDATION_F_SQUAD
        elif self.config["dataset"] == "squad2_transformers":
            dataset = Squad2Dataset_transformers
            in_train = TRAIN_F_SQUAD2
            in_dev = VALIDATION_F_SQUAD2
        elif self.config["dataset"] == "mrqa_transformers":
            dataset = MRQADataset_transformers
            dataset_dev = MRQADataset_transformers
            if self.config["subdataset"] == "squad":
                in_train = TRAIN_F_MRQA_SQUAD
                in_dev = VALIDATION_F_SQUAD
                dataset_dev = SquadDataset_transformers
            elif self.config["subdataset"] == "nq":
                in_train = TRAIN_F_MRQA_NQ
                in_dev = VALIDATION_F_MRQA_NQ
            elif self.config["subdataset"] == "newsqa":
                in_train = TRAIN_F_MRQA_NEWSQA
                in_dev = VALIDATION_F_MRQA_NEWSQA
            elif self.config["subdataset"] == "triviaqa":
                in_train = TRAIN_F_MRQA_TRIVIAQA
                in_dev = VALIDATION_F_MRQA_TRIVIAQA
            elif self.config["subdataset"] == "searchqa":
                in_train = TRAIN_F_MRQA_SEARCHQA
                in_dev = VALIDATION_F_MRQA_SEARCHQA
        else:
            raise NotImplementedError(f"Unknown dataset {self.config['dataset']}")

        fields = dataset.prepare_fields(pad_t=self.tokenizer.pad_token_id)

        if not self.config["validate_only"]:
            train = dataset(in_train, self.tokenizer, fields, transformer=self.config["model_type"],
                            is_training=True)
            logging.info(f"Training data size: {len(train.examples)}")
        val = dataset(in_dev, self.tokenizer, fields, transformer=self.config["model_type"], is_training=False)
        logging.info(f"Validation data size: {len(val.examples)}")

        if not self.config["validate_only"]:
            train_iter = BucketIterator(train, sort_key=lambda x: -(len(x.input)),
                                        shuffle=True, sort=False,
                                        batch_size=self.config["batch_size"], train=True,
                                        repeat=False,
                                        device=self.device)

        val_iter = BucketIterator(val, sort_key=lambda x: (len(x.input)), sort=True,
                                  batch_size=self.config["validation_batch_size"],
                                  repeat=False, shuffle=False,
                                  device=self.device)
        logging.info("Creating/Loading model")
        if not self.config["validate_only"]:
            model = self.create_model(self.config, self.device, self.tokenizer)
        else:
            model = self.load_model(self.config["model_to_validate"])
            if "no_ans_model" in self.config:
                model.no_ans_model = self.load_model(self.config["no_ans_model"])

        if self.config["multi_gpu"]:
            model = DataParallel(model)
            logging.info(f"Using device ids: {model.device_ids}")

        logging.info(f"Models has {count_parameters(model)} parameters")
        param_sizes, param_shapes = report_parameters(model)
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        if not self.config["validate_only"]:
            t_total = (len(train_iter.data()) // train_iter.batch_size + 1) // (
                    self.config["true_batch_size"] // self.config["batch_size"]) * self.config["epochs"]

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.config["weight_decay"],
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            optimizer = transformers.AdamW(optimizer_grouped_parameters,
                                           lr=self.config["learning_rate"],
                                           eps=self.config["adam_eps"])
            warmup_steps = round(self.config["warmup_proportion"] * t_total) if "warmup_proportion" in self.config else \
                self.config["warmup_steps"]
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=t_total
            )
            logging.info(f"Scheduler: warmup steps: {warmup_steps}, total_steps: {t_total}")

        start_time = time.time()
        try:
            best_val_loss = math.inf
            best_val_f1 = 0
            best_em = 0
            for epoch in range(self.config["epochs"]):
                self.epoch = epoch
                if not self.config["validate_only"]:
                    logging.info(f"Epoch {epoch}")
                    self.train_epoch(model, optimizer, scheduler=scheduler, train_iter=train_iter)
                # validation_loss, em, f1 = 0, 0, 0,
                validation_loss, em, f1 = self.validate(model, val_iter)
                logging.info(f"Validation loss/EM/F1: {validation_loss:.5f}, {em:.3f}, {f1:.3f}")

                if self.config["validate_only"]:
                    return
                if validation_loss < best_val_loss: best_val_loss = validation_loss
                if f1 > best_val_f1: best_val_f1 = f1
                if em > best_em:
                    best_em = em

                    if not self.config["save_only_last_epoch"]:
                        # Do all this on CPU, this is memory exhaustive!
                        model.to(torch.device("cpu"))
                        torch.save(model,
                                   f"{self.config['save_dir']}/checkpoint"
                                   f"_{self.config['dataset']}"
                                   f"_{self.config['model_type']}"
                                   f"_{self.config['objective_variant']}"
                                   f"_{str(self.__class__)}"
                                   f"_EM_{em:.2f}_F1_{f1:.2f}_L_{validation_loss:.2f}_{get_timestamp()}"
                                   f"_{socket.gethostname()}.pt")
                        model.to(self.device)
                if self.config["save_only_last_epoch"] and epoch == self.config["epochs"] - 1:
                    model.to(torch.device("cpu"))
                    torch.save(model,
                               f"{self.config['save_dir']}/checkpoint"
                               f"_{self.config['dataset']}"
                               f"_{self.config['model_type']}"
                               f"_{self.config['objective_variant']}"
                               f"_{str(self.__class__)}"
                               f"_EM_{em:.2f}_F1_{f1:.2f}_L_{validation_loss:.2f}_{get_timestamp()}"
                               f"_{socket.gethostname()}.pt")
                    model.to(self.device)
                logging.info(f"BEST L/F1/EM = {best_val_loss:.2f}/{best_val_f1:.2f}/{best_em:.2f}")
        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')

    def load_model(self, model_path):
        model = torch.load(model_path)
        if type(model) == DataParallel:
            model = model.module
        elif type(model) == OrderedDict:
            model_dict = model
            model = self.create_model(self.config, self.device, self.tokenizer)
            model.load_state_dict(model_dict)
        elif type(model) == dict:
            model_dict = model
            try:
                model = self.create_model(model_dict["config"], self.device, self.tokenizer)
            except BaseException as e:
                logging.error(e)
                model = self.create_model(self.config, self.device, self.tokenizer)
            model.load_state_dict(model_dict["state_dict"])
        model = model.to(self.device)

        convert_older = hasattr(model, "bert") or \
                        not hasattr(model, 'mask_cls') or \
                        not hasattr(model, "pad_token_id") or \
                        hasattr(model, "topk") or \
                        not hasattr(model, "beam_search_topn")

        if convert_older:
            # Convert older checkpoints
            if not hasattr(model, 'mask_cls'):
                model.mask_cls = not (
                        self.config["dataset"] in ["squad_transformers", "mrqa_transformers", "squad2_transformers"])
            if hasattr(model, "bert"):
                model.transformer = model.bert
            if not hasattr(model, "pad_token_id"):
                model.pad_token_id = self.tokenizer.pad_token_id
            if hasattr(model, "topk"):
                model.beam_search_topn = model.topk
            if not hasattr(model, "beam_search_topn"):
                model.beam_search_topn = 10  # hardcoded for now

        return model

    @staticmethod
    def create_model(config, device, tokenizer):
        keep_cls_o = config["dataset"] in ["squad_transformers",
                                           "mrqa_transformers",
                                           "squad2_transformers"]
        mask_cls = False  # do not mask cls during training, set this attribute explicitly in test time
        if config["objective_variant"] == "independent":
            if config["model_type"].startswith("albert"):
                model = AlbertForQuestionAnswering.from_pretrained(config["model_type"],
                                                                   cache_dir=config["cache_dir"])
            else:
                model = BertForQuestionAnswering.from_pretrained(config["model_type"],
                                                                 cache_dir=config["cache_dir"])
        elif config["objective_variant"] == "joint":
            inverse_segment_mask = config["dataset"] in ["squad_transformers", "squad2_transformers", "mrqa_transformers"]
            if config["model_type"].startswith("albert"):
                model_clz = AlbertModel
            elif config["model_type"].startswith("bert"):
                model_clz = BertModel
            else:
                raise NotImplementedError(f"Unknown model type: '{config['model_type']}'")
            model = TransformerForQuestionAnsweringJoint(config,
                                                         model_clz=model_clz,
                                                         inverse_segment_mask=inverse_segment_mask,
                                                         mask_cls=mask_cls,
                                                         sep_token_id=tokenizer.sep_token_id,
                                                         pad_token_id=tokenizer.pad_token_id,
                                                         keep_cls_level_output=keep_cls_o)
        elif config["objective_variant"] == "independent_joint":
            inverse_segment_mask = config["dataset"] in ["squad_transformers", "mrqa_transformers",
                                                         "squad2_transformers"]

            if config["model_type"].startswith("albert"):
                model_clz = AlbertModel
            elif config["model_type"].startswith("bert"):
                model_clz = BertModel
            else:
                raise NotImplementedError(f"Unknown model type: '{config['model_type']}'")

            model = TransformerForQuestionAnsweringIndepJoint(config,
                                                              model_clz=model_clz,
                                                              inverse_segment_mask=inverse_segment_mask,
                                                              mask_cls=mask_cls,
                                                              sep_token_id=tokenizer.sep_token_id,
                                                              pad_token_id=tokenizer.pad_token_id,
                                                              keep_cls_level_output=keep_cls_o)
        elif config["objective_variant"] == "conditional":
            inverse_segment_mask = config["dataset"] in ["squad_transformers", "squad2_transformers","mrqa_transformers"]

            if config["model_type"].startswith("albert"):
                model_clz = AlbertModel
            elif config["model_type"].startswith("bert"):
                model_clz = BertModel
            else:
                raise NotImplementedError(f"Unknown model type: '{config['model_type']}'")

            model = TransformerForQuestionAnsweringConditional(config,
                                                               model_clz=model_clz,
                                                               inverse_segment_mask=inverse_segment_mask,
                                                               mask_cls=mask_cls,
                                                               sep_token_id=tokenizer.sep_token_id,
                                                               pad_token_id=tokenizer.pad_token_id,
                                                               keep_cls_level_output=keep_cls_o)
        model = model.to(device)
        model.config = config
        return model

    def log_results_squad11(self, results, scores, diffs=None, val_file=".data/squad/dev-v1.1.json"):
        f = open(f"results/result_{get_timestamp()}_{socket.gethostname()}.csv", mode="w")
        csvw = csv.writer(f, delimiter=',')
        HEADER = ["Correct", "Ground Truth(s)", "Prediction", "Confidence", "Question", "Context", "Topic", "ID"]
        if diffs is not None:
            HEADER = ["Diff"] + HEADER
        csvw.writerow(HEADER)
        with open(val_file) as fd:
            data_json = json.load(fd)
            for data_topic in data_json["data"]:
                for paragraph in data_topic["paragraphs"]:
                    for question_and_answers in paragraph['qas']:
                        prediction = results[question_and_answers["id"]]
                        confidence = str(f"{scores[question_and_answers['id']]:.2f}")
                        answers = "|".join(map(lambda x: x['text'], question_and_answers['answers']))
                        correct = int(results[question_and_answers["id"]].lower() in map(lambda x: x['text'].lower(),
                                                                                         question_and_answers[
                                                                                             'answers'])) or int(
                            answers == prediction)
                        ex = [correct,
                              answers,
                              prediction,
                              confidence,
                              question_and_answers['question'],
                              paragraph["context"],
                              data_topic["title"],
                              question_and_answers["id"]]
                        if diffs is not None:
                            ex = [diffs[question_and_answers["id"]]] + ex
                        csvw.writerow(ex)
        f.close()
