import json
import logging
import os
import pickle
import shutil
import string
import sys
import time
import torchtext.data as data

from typing import List, Tuple, Dict
from torchtext.data import RawField, Example
from transformers import SquadV1Processor, squad_convert_examples_to_features, PreTrainedTokenizer, SquadFeatures
from src.datasets.utility import TRAIN_F_SQUAD, VALIDATION_F_SQUAD, download_url, TRAIN_V1_URL_SQUAD, \
    DEV_V1_URL_SQUAD


class SquadDataset_transformers(data.Dataset):
    def __init__(self, data_f: string, tokenizer: PreTrainedTokenizer, fields: List[Tuple[str, data.Field]],
                 transformer,
                 is_training,
                 skip_no_answer_windows=False,
                 cachedir='.data/squad', max_len=512 - 3, **kwargs):
        self.check_for_download(cachedir)

        self.max_len = max_len
        self.skip_no_ans_w = skip_no_answer_windows

        self.cachedir = cachedir
        self.data_f = data_f
        self.tokenizer = tokenizer
        self.is_training = is_training
        preprocessed_f_noext = os.path.join(cachedir, data_f) + f"_preprocessed_for_{transformer}_transformers"
        preprocessed_f = preprocessed_f_noext + ".json"
        if not os.path.exists(preprocessed_f):
            s_time = time.time()
            transformers_examples, transformers_features, examples = self.get_example_list()
            self.save(preprocessed_f, examples)
            # in transformers, also save problem examples/features
            self.save_pkl(preprocessed_f_noext + "_problem.pkl", transformers_examples)
            self.save_pkl(preprocessed_f_noext + "_features.pkl", transformers_features)
            logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")
        s_time = time.time()
        examples = self.load(preprocessed_f, fields)
        self.problem_examples = self.load_pkl(preprocessed_f_noext + "_problem.pkl")
        self.problem_features = self.load_pkl(preprocessed_f_noext + "_features.pkl")
        logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")

        super(SquadDataset_transformers, self).__init__(examples, fields, **kwargs)

    def save_pkl(self, file, obj):
        with open(file, "wb") as f:
            pickle.dump(obj, f)

    def load_pkl(self, file):
        with open(file, "rb") as f:
            return pickle.load(f)

    def save(self, preprocessed_f: string, raw_examples: List[Dict]):
        with open(preprocessed_f, "w") as f:
            json.dump(raw_examples, f)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with open(preprocessed_f, "r") as f:
            raw_examples = json.load(f)
            return [data.Example.fromlist([
                e["id"],
                e["input"],
                e["segment_ids"],
                e["input_padding"],
                e["a_start"],
                e["a_end"],
                e["token_is_max_context"],
                e["paragraph_mask"],
                e["example_idx"],
                e["token_to_orig_map"],
            ], fields) for e in raw_examples]

    @classmethod
    def splits(cls, tokenizer, fields, cachedir='.data/squad'):
        cls.check_for_download(cachedir)
        train_data = cls(TRAIN_F_SQUAD, tokenizer, fields, cachedir=cachedir)
        val_data = cls(VALIDATION_F_SQUAD, tokenizer, fields, cachedir=cachedir)
        return tuple(d for d in (train_data, val_data)
                     if d is not None)

    @staticmethod
    def check_for_download(cachedir):
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
            try:
                download_url(os.path.join(cachedir, TRAIN_F_SQUAD), TRAIN_V1_URL_SQUAD)
                download_url(os.path.join(cachedir, VALIDATION_F_SQUAD), DEV_V1_URL_SQUAD)
            except BaseException as e:
                sys.stderr.write(f'Download failed, removing directory {cachedir}\n')
                sys.stderr.flush()
                shutil.rmtree(cachedir)
                raise e

    def get_example_list(self):
        logging.info("Running transformers data preprocessing...")
        processor = SquadV1Processor()
        if self.is_training:
            examples = processor.get_train_examples(self.cachedir, filename=self.data_f)
        else:
            examples = processor.get_dev_examples(self.cachedir, filename=self.data_f)
        max_seq_length = 384
        doc_stride = 128
        max_query_length = 64
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_training=self.is_training,
            return_dataset=False,
            threads=4,
        )
        torchtext_examples = []
        for feature in features:
            feature: SquadFeatures = feature
            if self.is_training and self.skip_no_ans_w:
                if feature.start_position == 0 and feature.end_position == 0:
                    continue
            torchtext_examples.append({"id": feature.unique_id,
                                       "input": feature.input_ids,
                                       "segment_ids": feature.token_type_ids,
                                       "input_padding": feature.attention_mask,
                                       "a_start": feature.start_position,
                                       "a_end": feature.end_position,
                                       "token_is_max_context": feature.token_is_max_context,
                                       "paragraph_mask": feature.p_mask,
                                       "example_idx": feature.example_index,
                                       "token_to_orig_map": feature.token_to_orig_map})
        return examples, features, torchtext_examples

    @staticmethod
    def prepare_fields(pad_t):
        WORD_field = data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=pad_t)
        return [
            ('id', data.RawField()),
            ('input', WORD_field),
            ('segment_ids', data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ('input_padding', data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=0)),
            ("a_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("a_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('is_max_context', data.RawField()),
            ('paragraph_mask', data.Field(use_vocab=False, batch_first=True, sequential=True, pad_token=1)),
            ('example_idx', data.RawField()),
            ('token_to_orig_map', data.RawField()),
        ]
