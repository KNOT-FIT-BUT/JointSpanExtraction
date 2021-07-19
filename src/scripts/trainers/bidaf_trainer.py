import csv
import json
import math
import os
import sys
import socket
import time
import logging
import traceback

import torch
import torchtext

from collections import defaultdict
from typing import Tuple

from torch.nn import DataParallel
from torch.nn.modules.loss import _Loss, CrossEntropyLoss
from torch.optim import Adadelta, Adam
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.datasets.utility import VALIDATION_F_SQUAD
from torchtext.data import BucketIterator, Iterator
from torchtext.vocab import GloVe
from src.datasets.SQUAD_dataset_BIDAF_QA import SquadDatasetBidaf
from src.models.bidaf.bidaf_compound import BidAF_compound
from src.models.bidaf.bidaf_joint import BidAF_joint
from src.models.bidaf.bidaf_vanilla import BidAF
from src.models.utils.ema import EMA
from src.models.utils.modelparts import SpanPredictionModule
from src.scripts.tools.evaluate_squad_1_1 import evaluate_squad_11
from src.scripts.tools.utility import count_parameters, report_parameters, get_timestamp, mkdir


def get_model(m):
    if type(m) == DataParallel:
        return m.module
    return m


class BidafModelFramework:
    def __init__(self, config, device):
        mkdir(config["result_dir"])
        mkdir(config["save_dir"])
        self.config = config
        self.save_threshold = config["min_em_to_save"]
        self.device = device

    def train_epoch(self, model: torch.nn.Module, lossfunction: _Loss, optimizer: torch.optim.Optimizer, scheduler,
                    train_iter: Iterator) -> float:
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        update_ratio = self.config["true_batch_size"] // self.config["batch_size"]
        updated = False
        total_its = len(train_iter.data()) // train_iter.batch_size + 1
        it = tqdm(enumerate(train_iter), total=total_its)
        for i, batch in it:
            if self.config["objective_variant"] == "independent":
                logprobs_S, logprobs_E = model(batch.document, batch.question, batch.document_char, batch.question_char,
                                               mask_rnns=self.config["mask_rnns"])
                loss_s = lossfunction(logprobs_S, batch.a_start)
                loss_e = lossfunction(logprobs_E, batch.a_end)
                loss = (loss_s + loss_e) / 2.
            elif self.config["objective_variant"] == "joint":
                try:
                    logprobs_J = model(batch.document, batch.question, batch.document_char, batch.question_char,
                                       mask_rnns=self.config["mask_rnns"])
                except RuntimeError as re:
                    torch.cuda.empty_cache()
                    logging.error(re)
                    tb = traceback.format_exc()
                    logging.error(tb)
                    continue
                loss = lossfunction(logprobs_J.view(logprobs_J.shape[0], -1),
                                    batch.a_start * batch.document[0].shape[-1] + batch.a_end)
            elif self.config["objective_variant"] == "independent_joint":
                logprobs_S, logprobs_E, logprobs_J = model(batch.document, batch.question, batch.document_char,
                                                           batch.question_char, mask_rnns=self.config["mask_rnns"])
                loss_s = lossfunction(logprobs_S, batch.a_start)
                loss_e = lossfunction(logprobs_E, batch.a_end)
                loss_joint = lossfunction(logprobs_J.view(logprobs_J.shape[0], -1),
                                          batch.a_start * logprobs_E.shape[-1] + batch.a_end)
                loss = (loss_s + loss_e + loss_joint) / 3.

            loss = loss / update_ratio
            loss.backward()
            if (i + 1) % update_ratio == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 2.)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                updated = True
                it.set_description(f"Training loss: {train_loss / (i + 1e-10) + 1}")

            train_loss += loss.item()

        # Do the last step if needed with what has been accumulated
        if not updated:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        return train_loss / len(train_iter.data())

    @staticmethod
    def get_spans(batch, candidates):
        r = []
        for i in range(len(batch.raw_document_context)):
            candidate_start = candidates[0][i]
            candidates_end = candidates[1][i]
            if candidate_start > len(batch.document_token_positions[i]) - 1:
                candidate_start = len(batch.document_token_positions[i]) - 1
            if candidates_end > len(batch.document_token_positions[i]) - 1:
                candidates_end = len(batch.document_token_positions[i]) - 1

            r.append(batch.raw_document_context[i][batch.document_token_positions[i][candidate_start][0]:
                                                   batch.document_token_positions[i][candidates_end][-1]])
        return r

    @torch.no_grad()
    def validate(self, model: torch.nn.Module, lossfunction: _Loss, iter: Iterator, ema=None, log_results=False) -> \
            Tuple[float, float, float]:
        model.eval()
        if ema is not None:
            backup_params = EMA.ema_backup_and_loadavg(ema, get_model(model))

        results = dict()
        ids = []
        lossvalues = []
        spans = []
        gt_spans = []
        span_probs = []
        it = tqdm(iter, total=len(iter.data()) // iter.batch_size + 1)
        for i, batch in enumerate(it):
            ids += batch.id
            inputs = batch.document, batch.question, batch.document_char, batch.question_char
            try:
                outputs = model(*inputs, mask_rnns=self.config["mask_rnns"])
            except TypeError as e:
                logging.error(e)
                logging.error(
                    "Probably encountered a dataparallel error, see https://github.com/pytorch/pytorch/issues/15161 for more details.")
                logging.error("Retrying without dataparallel...")
                outputs = get_model(model)(*inputs, mask_rnns=self.config["mask_rnns"])

            if self.config["objective_variant"] == "independent":
                logprobs_S, logprobs_E = outputs
                loss_s = lossfunction(logprobs_S, batch.a_start)
                loss_e = lossfunction(logprobs_E, batch.a_end)
                loss = (loss_s + loss_e) / 2.
            elif self.config["objective_variant"] == "joint":
                logprobs_J = outputs
                loss = lossfunction(logprobs_J.view(logprobs_J.shape[0], -1),
                                    batch.a_start * batch.document[0].shape[-1] + batch.a_end)
            elif self.config["objective_variant"] == "independent_joint":
                logprobs_S, logprobs_E, logprobs_J = outputs
                loss_s = lossfunction(logprobs_S, batch.a_start)
                loss_e = lossfunction(logprobs_E, batch.a_end)
                loss_joint = lossfunction(logprobs_J.view(logprobs_J.shape[0], -1),
                                          batch.a_start * logprobs_E.shape[-1] + batch.a_end)
                loss = (loss_s + loss_e + loss_joint) / 3.

            lossvalues += loss.tolist()

            if self.config["decoding_variant"] == "independent":
                best_span_probs, candidates = SpanPredictionModule.decode(logprobs_S, logprobs_E)
            elif self.config["decoding_variant"] == "independent_filtered":
                best_span_probs, candidates = SpanPredictionModule.decode_wth_hacks(logprobs_S, logprobs_E, hacks={
                    "max_answer_length": self.config["max_answer_length"]
                })
            elif self.config["decoding_variant"] == "independent_filtered_combined":
                best_span_probs, candidates = SpanPredictionModule.decode_wth_hacks(logprobs_S, logprobs_E,score="probs", hacks={
                    "max_answer_length": self.config["max_answer_length"],
                    "combine_surface_forms": (
                        True, batch.document if not type(batch.document) == tuple else batch.document[0], iter.dataset.fields["question"].vocab.stoi["<pad>"])
                })
            elif self.config["decoding_variant"] == "joint":
                best_span_probs, candidates = SpanPredictionModule.decode_joint(logprobs_J)

            elif self.config["decoding_variant"] == "joint_filtered":
                best_span_probs, candidates = SpanPredictionModule.decode_joint_with_hacks(logprobs_J, hacks={
                    "max_answer_length": self.config["max_answer_length"]
                })
            elif self.config["decoding_variant"] == "joint_filtered_combined":
                best_span_probs, candidates = SpanPredictionModule.decode_joint_with_hacks(logprobs_J, score="probs",hacks={
                    "max_answer_length": self.config["max_answer_length"],
                    "combine_surface_forms": (
                        True, batch.document if not type(batch.document) == tuple else batch.document[0], iter.dataset.fields["question"].vocab.stoi["<pad>"])
                })
            else:
                raise ValueError(f"Unknown decoding variant {self.config['decoding_variant']}")

            span_probs += best_span_probs.tolist()
            spans += self.get_spans(batch, candidates)

            gt_spans += batch.gt_answer

        # compute the final loss and results
        # we need to filter trhough multiple possible choices and pick the best one
        lossdict = defaultdict(lambda: math.inf)
        probs = defaultdict(lambda: 0)
        for id, value, span, span_prob in zip(ids, lossvalues, spans, span_probs):
            # record only lowest loss
            if lossdict[id] > value:
                lossdict[id] = value
            results[id] = span
            probs[id] = span_prob

        if log_results:
            self.log_results(results, probs)

        loss = sum(lossdict.values()) / len(lossdict)
        prediction_file = f".data/squad/dev_results_{socket.gethostname()}.json"
        with open(prediction_file, "w") as f:
            json.dump(results, f)

        dataset_file = ".data/squad/dev-v1.1.json"

        expected_version = '1.1'
        with open(dataset_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            if (dataset_json['version'] != expected_version):
                logging.info('Evaluation expects v-' + expected_version +
                             ', but got dataset with v-' + dataset_json['version'],
                             file=sys.stderr)
            dataset = dataset_json['data']
        with open(prediction_file) as prediction_file:
            predictions = json.load(prediction_file)
        result = evaluate_squad_11(dataset, predictions)
        logging.info(json.dumps(result))

        if ema is not None:
            EMA.ema_restore_backed_params(backup_params, get_model(model))

        return loss, result["exact_match"], result["f1"]

    @staticmethod
    def load_data(config):
        if config["char_embeddings"]:
            fields = SquadDatasetBidaf.prepare_fields_char(
                use_start_end_char_boundaries=config["use_start_end_char_boundaries"],
                include_lengths=config["mask_rnns"])
        else:
            fields = SquadDatasetBidaf.prepare_fields()
        if not config["validate_only"]:
            train, val = SquadDatasetBidaf.splits(fields)
            logging.info(f"Training data size: {len(train.examples)}")
        else:
            val = SquadDatasetBidaf(VALIDATION_F_SQUAD, fields)
        logging.info(f"Validation data size: {len(val.examples)}")
        fields = dict(fields)
        train = None
        if not config["validate_only"]:
            fields["question"].build_vocab(train, val, vectors=GloVe(name='6B', dim=config["embedding_size"]))

            if not type(fields["question_char"]) == torchtext.data.field.RawField:
                fields["question_char"].build_vocab(train, val, max_size=config["char_maxsize_vocab"])
        return train, val, fields

    def fit(self, train=None, val=None, fields=None):
        device = self.device
        config = self.config
        logging.debug(json.dumps(config, indent=4, sort_keys=True))

        if (not self.config["validate_only"] and None in (train, val, fields)) or (
                self.config["validate_only"] and None in (val, fields)):
            train, val, fields = BidafModelFramework.load_data(config)

        if not self.config["validate_only"]:
            train_iter = BucketIterator(train, sort_key=lambda x: -(len(x.question) + len(x.document)),
                                        shuffle=True, sort=False, sort_within_batch=True,
                                        batch_size=config["batch_size"], train=True,
                                        repeat=False,
                                        device=device)

            q_vocab = fields['question'].vocab
            q_char_vocab = fields["question_char"].vocab
            model = self.create_model(config, device, q_vocab, q_char_vocab)

        val_iter = BucketIterator(val, sort_key=lambda x: -(len(x.question) + len(x.document)), sort=True,
                                  batch_size=config["batch_size"],
                                  repeat=False,
                                  device=device)
        if self.config["validate_only"]:
            saved_model = torch.load(self.config["model_path"], map_location=device)
            if type(saved_model) == dict:
                model = self.create_model(config, device, saved_model["token_vocab"], saved_model["char_token_vocab"])
                model.load_state_dict(saved_model["state_dict"])
            else:
                model = saved_model
            fields["question"].vocab = model.embedder.vocab
            fields["question_char"].nesting_field.vocab = model.char_embedder.vocab

            model.encoder.rnn.flatten_parameters()
            model.modeling_layer.rnn.flatten_parameters()
            model.output_layer.rnn.flatten_parameters()

        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
            logging.info(f"Using device ids: {model.device_ids}")
        model = model.to(device)

        logging.info(f"Models has {count_parameters(model)} parameters")
        param_sizes, param_shapes = report_parameters(get_model(model))
        param_sizes = "\n'".join(str(param_sizes).split(", '"))
        param_shapes = "\n'".join(str(param_shapes).split(", '"))
        logging.debug(f"Model structure:\n{param_sizes}\n{param_shapes}\n")

        if not self.config["validate_only"]:

            if config["optimizer"] == "adam":
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=config["learning_rate"])
            elif config["optimizer"] == "adadelta":
                optimizer = Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=config["learning_rate"])
            else:
                raise NotImplementedError(f"Option {config['optimizer']} for \"optimizer\" setting is undefined.")

            if self.config["scheduler"] == "linear":
                t_total = (len(train_iter.data()) // train_iter.batch_size + 1) // (
                        self.config["true_batch_size"] // self.config["batch_size"]) * self.config["epochs"]
                warmup_steps = round(
                    self.config["warmup_proportion"] * t_total) if "warmup_proportion" in self.config else \
                    self.config["warmup_steps"]
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=t_total
                )
                logging.info(f"Scheduler: warmup steps: {warmup_steps}, total_steps: {t_total}")
            else:
                scheduler = None

        best_em = 0
        best_ckpt = None
        best_val_f1 = 0
        best_val_loss = math.inf
        start_time = time.time()
        try:
            ema_active = False
            no_improvement_steps = 0
            for it in range(config["epochs"]):
                logging.info(f"Epoch {it}")

                if not self.config["validate_only"]:
                    if "ema" in config and config["ema"]:
                        ema = EMA.ema_register(config, model)
                        ema_active = True
                    self.train_epoch(model, CrossEntropyLoss(), optimizer, scheduler, train_iter)

                    if ema_active:
                        EMA.ema_update(ema, model)

                validation_loss, em, f1 = self.validate(model, CrossEntropyLoss(reduction='none'), val_iter,
                                                        ema=ema if "ema" in config and config[
                                                            "ema"] and ema_active and not self.config[
                                                            "validate_only"] else None)
                if not self.config["validate_only"]:
                    if validation_loss < best_val_loss: best_val_loss = validation_loss
                    if f1 > best_val_f1: best_val_f1 = f1
                    if em > best_em:
                        no_improvement_steps = 0
                        best_em = em
                        if em > self.save_threshold:
                            # Do all this on CPU, this is memory exhaustive!
                            model.to(torch.device("cpu"))
                            best_ckpt = os.path.join(self.config["save_dir"],
                                                     f"checkpoint" \
                                                     f"_{str(self.__class__)}" \
                                                     f"_EM_{em:.2f}_F1_{f1:.2f}_L_{validation_loss:.2f}_{get_timestamp()}" \
                                                     f"_{socket.gethostname()}.pt")
                            if ema_active:
                                # backup current params and load ema params
                                backup_params = EMA.ema_backup_and_loadavg(ema, get_model(model))

                                torch.save(get_model(model), best_ckpt)

                                # load back backed up params
                                EMA.ema_restore_backed_params(backup_params, get_model(model))

                            else:
                                torch.save(get_model(model), best_ckpt)

                            model.to(device)
                    else:
                        no_improvement_steps += 1

                logging.info(f"BEST L/F1/EM = {best_val_loss:.2f}/{best_val_f1:.2f}/{best_em:.2f}")
                logging.info(f"Validation loss: {validation_loss}")
                if self.config["validate_only"]:
                    return
                if no_improvement_steps > config["patience"]:
                    logging.info(f"Running out of patience with em {best_em:.5f}")
                    break

        except KeyboardInterrupt:
            logging.info('-' * 120)
            logging.info('Exit from training early.')
        finally:
            logging.info(f'Finished after {(time.time() - start_time) / 60} minutes.')
        return best_em, best_ckpt

    def create_model(self, config, device, q_vocab, q_char_vocab):
        logging.info(f"Creating model: {self.config['objective_variant']}")
        if self.config["objective_variant"] == "independent":
            model = BidAF(config, q_vocab, q_char_vocab)
        elif self.config["objective_variant"] == "joint":
            model = BidAF_joint(config, q_vocab, q_char_vocab)
        elif self.config["objective_variant"] == "independent_joint":
            model = BidAF_compound(config, q_vocab, q_char_vocab)
        return model.to(device)

    def predict(self, modelname, inpf, outf):
        logging.debug(json.dumps(self.config, indent=4, sort_keys=True))
        fields = SquadDatasetBidaf.prepare_fields_char(
            use_start_end_char_boundaries=self.config["use_start_end_char_boundaries"],
            include_lengths=self.config["mask_rnns"])

        model = torch.load(modelname, map_location=self.device)
        model.encoder.rnn.flatten_parameters()
        model.modeling_layer.rnn.flatten_parameters()
        model.output_layer.rnn.flatten_parameters()
        model.eval()
        fields = dict(fields)
        fields["question"].vocab = model.embedder.vocab
        fields["question_char"].nesting_field.vocab = model.char_embedder.vocab

        test = SquadDatasetBidaf(inpf, list(fields.items()))

        test_iter = BucketIterator(test, sort_key=lambda x: -(len(x.question) + len(x.document)), sort=True,
                                   batch_size=self.config["validation_batch_size"],
                                   repeat=False,
                                   device=self.device)

        results = dict()
        ids = []
        spans = []
        gt_spans = []
        span_probs = []
        it = tqdm(test_iter, total=len(test_iter.data()) // test_iter.batch_size + 1)
        for i, batch in enumerate(it):
            ids += batch.id
            if self.config["objective_variant"] == "independent":
                logprobs_S, logprobs_E = model(batch.document, batch.question, batch.document_char, batch.question_char,
                                               mask_rnns=self.config["mask_rnns"])
            elif self.config["objective_variant"] == "joint":
                logprobs_J = model(batch.document, batch.question, batch.document_char, batch.question_char,
                                   mask_rnns=self.config["mask_rnns"])
            elif self.config["objective_variant"] == "independent_joint":
                logprobs_S, logprobs_E, logprobs_J = model(batch.document, batch.question, batch.document_char,
                                                           batch.question_char, mask_rnns=self.config["mask_rnns"])

            if self.config["decoding_variant"] == "independent":
                best_span_probs, candidates = BidAF.decode(logprobs_S, logprobs_E)
            elif self.config["decoding_variant"] == "independent_filtered":
                best_span_probs, candidates = BidAF.decode_wth_hacks(logprobs_S, logprobs_E, hacks={
                    "max_answer_length": self.config["max_answer_length"]
                })
            elif self.config["decoding_variant"] == "independent_filtered_combined":
                best_span_probs, candidates = BidAF.decode_wth_hacks(logprobs_S, logprobs_E, hacks={
                    "max_answer_length": self.config["max_answer_length"],
                    "combine_surface_forms": (
                        True, batch.document if not type(batch.document) == tuple else batch.document[0])
                })

            elif self.config["decoding_variant"] == "joint":
                best_span_probs, candidates = BidAF.decode_joint(logprobs_J)

            elif self.config["decoding_variant"] == "joint_filtered":
                best_span_probs, candidates = BidAF.decode_joint_with_hacks(logprobs_J, hacks={
                    "max_answer_length": self.config["max_answer_length"]
                })
            elif self.config["decoding_variant"] == "joint_filtered_combined":
                best_span_probs, candidates = BidAF.decode_joint_with_hacks(logprobs_J, hacks={
                    "max_answer_length": self.config["max_answer_length"],
                    "combine_surface_forms": (
                        True, batch.document if not type(batch.document) == tuple else batch.document[0])
                })

            span_probs += best_span_probs.tolist()
            spans += self.get_spans(batch, candidates)

            gt_spans += batch.gt_answer
            # compute the final loss and results
            # we need to filter through multiple possible choices and pick the best one
        probs = defaultdict(lambda: 0)
        assert len(ids) == len(spans) == len(span_probs)
        for id, span, span_prob in zip(ids, spans, span_probs):
            results[id] = span
            probs[id] = span_prob

        with open(outf, "w") as f:
            json.dump(results, f)

    def log_results(self, results, probs, val_file=".data/squad/dev-v1.1.json"):
        f = open(os.path.join(self.config["result_dir"],
                              f"result_{get_timestamp()}_{socket.gethostname()}.csv"), mode="w")
        csvw = csv.writer(f, delimiter=',')
        HEADER = ["Correct", "Ground Truth(s)", "Prediction", "Confidence", "Question", "Context", "Topic", "ID"]
        csvw.writerow(HEADER)
        with open(val_file) as fd:
            data_json = json.load(fd)
            for data_topic in data_json["data"]:
                for paragraph in data_topic["paragraphs"]:
                    for question_and_answers in paragraph['qas']:
                        prediction = results[question_and_answers["id"]]
                        confidence = str(f"{probs[question_and_answers['id']]:.2f}")
                        answers = "|".join(map(lambda x: x['text'], question_and_answers['answers']))
                        correct = int(results[question_and_answers["id"]].lower() in map(lambda x: x['text'].lower(),
                                                                                         question_and_answers[
                                                                                             'answers']))
                        ex = [correct,
                              answers,
                              prediction,
                              confidence,
                              question_and_answers['question'],
                              paragraph["context"],
                              data_topic["title"],
                              question_and_answers["id"]]
                        csvw.writerow(ex)
        f.close()
