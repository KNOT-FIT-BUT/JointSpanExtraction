import csv
import json
import os
import shutil
import time
import string
import sys
import numpy as np
import torchtext.data as data
import logging

from typing import List, Tuple, Dict
from torchtext.data import RawField, Example
from src.datasets.utility import TRAIN_V1_URL_SQUAD, DEV_V1_URL_SQUAD, TRAIN_F_SQUAD, VALIDATION_F_SQUAD, JOIN_TOKEN, \
    download_url, make_sure_min_length, find_sub_list
from src.scripts.tools.spacy_tokenizer import tokenize, char_span_to_token_span, tokenize_and_join


class SquadDatasetBidaf(data.Dataset):

    def __init__(self, data, fields: List[Tuple[str, data.Field]], cachedir='.data/squad', compact=False, **kwargs):
        """
        :param data:
        :param fields:
        :param cachedir:
        :param compact: if this setting is True, squad validation data, where each question has 3 answers
               will be treated as one example, instead of three
        :param kwargs:
        """
        f = os.path.join(cachedir, data)
        preprocessed_f = f + "_preprocessed_bidaf.json" if not compact else f + "_preprocessed_bidaf_compact.json"
        if not os.path.exists(preprocessed_f):
            s_time = time.time()
            raw_examples = SquadDatasetBidaf.get_example_list(f, compact)
            self.save(preprocessed_f, raw_examples)
            logging.info(f"Dataset {preprocessed_f} created in {time.time() - s_time:.2f}s")

        s_time = time.time()
        examples = self.load(preprocessed_f, fields)
        logging.info(f"Dataset {preprocessed_f} loaded in {time.time() - s_time:.2f} s")

        super(SquadDatasetBidaf, self).__init__(examples, fields, **kwargs)

    def save(self, preprocessed_f: string, raw_examples: List[Dict]):
        with open(preprocessed_f, "w") as f:
            json.dump(raw_examples, f)

    def load(self, preprocessed_f: string, fields: List[Tuple[str, RawField]]) -> List[Example]:
        with open(preprocessed_f, "r") as f:
            raw_examples = json.load(f)
            return [data.Example.fromlist([
                e["id"],
                e["topic"],
                e["paragraph_token_positions"],
                e["raw_paragraph_context"],
                e["paragraph_context"],
                e["paragraph_context"],
                e["paragraph_context"],
                e["question"],
                e["question"],
                e["raw_question"],
                e["a_start"],
                e["a_end"],
                e["a_extracted"],
                e["a_gt"]
            ], fields) for e in raw_examples]

    @classmethod
    def splits(cls, fields, cachedir='.data/squad'):
        cls.check_for_download(cachedir)
        train_data = cls(TRAIN_F_SQUAD, fields, cachedir=cachedir)
        val_data = cls(VALIDATION_F_SQUAD, fields, cachedir=cachedir)
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

    @staticmethod
    def get_example_list(file, compact=False):
        examples = [] if not compact else dict()
        cnt = 0

        ## debug
        f = open(f".data/squad/debug_{os.path.basename(file)}.csv", "a+")
        problems = 0

        with open(file) as fd:
            data_json = json.load(fd)
            for data_topic in data_json["data"]:
                topic_title = data_topic["title"]
                for paragraph in data_topic["paragraphs"]:
                    paragraph_tokens, paragraph_context = tokenize(paragraph["context"])
                    paragraph_token_positions = [[token.idx, token.idx + len(token.text)] for token in paragraph_tokens]
                    joined_paragraph_context = JOIN_TOKEN.join(paragraph_context)
                    for question_and_answers in paragraph['qas']:
                        example_id = question_and_answers["id"]
                        question = tokenize_and_join(question_and_answers['question'])
                        answers = question_and_answers['answers']

                        for possible_answer in answers:
                            answer_start_ch = possible_answer["answer_start"]
                            answer_end = possible_answer["answer_start"] + len(possible_answer["text"])
                            answer_tokens, answer = tokenize(possible_answer["text"])

                            answer_locations = find_sub_list(answer, paragraph_context)
                            if len(answer_locations) > 1:
                                # get start character offset of each span
                                answer_ch_starts = [paragraph_tokens[token_span[0]].idx for token_span in
                                                    answer_locations]
                                distance_from_gt = np.abs((np.array(answer_ch_starts) - answer_start_ch))
                                closest_match = distance_from_gt.argmin()

                                answer_start, answer_end = answer_locations[closest_match]
                            elif not answer_locations:
                                # Call heuristic from AllenNLP to help :(
                                token_span = char_span_to_token_span(
                                    [(t.idx, t.idx + len(t.text)) for t in paragraph_tokens],
                                    (answer_start_ch, answer_end))
                                answer_start, answer_end = token_span[0]
                            else:
                                answer_start, answer_end = answer_locations[0]
                            cnt += 1

                            ## Debug
                            def is_correct():
                                def remove_ws(s):
                                    return "".join(s.split())

                                csvf = csv.writer(f, delimiter=',')
                                if remove_ws(possible_answer["text"]) != remove_ws(
                                        "".join(paragraph_context[answer_start:answer_end + 1])):
                                    csvf.writerow({"id": example_id,
                                                   "topic": topic_title,
                                                   "raw_paragraph_context": paragraph["context"],
                                                   "paragraph_context": joined_paragraph_context,
                                                   "paragraph_token_positions": paragraph_token_positions,
                                                   "question": question,
                                                   "a_start": answer_start,
                                                   "a_end": answer_end,
                                                   "a_extracted": JOIN_TOKEN.join(
                                                       paragraph_context[answer_start:answer_end + 1]),
                                                   "a_gt": possible_answer["text"]}.values())
                                    return False
                                return True

                            if not is_correct():
                                problems += 1
                            if not compact:
                                examples.append({"id": example_id,
                                                 "topic": topic_title,
                                                 "raw_paragraph_context": paragraph["context"],
                                                 "paragraph_context": joined_paragraph_context,
                                                 "paragraph_token_positions": paragraph_token_positions,
                                                 "raw_question": question_and_answers['question'],
                                                 "question": question,
                                                 "a_start": answer_start,
                                                 "a_end": answer_end,
                                                 "a_extracted": JOIN_TOKEN.join(
                                                     paragraph_context[answer_start:answer_end + 1]),
                                                 "a_gt": possible_answer["text"]})
                            else:
                                if not example_id in examples:
                                    examples[example_id] = {"id": example_id,
                                                            "topic": topic_title,
                                                            "raw_paragraph_context": paragraph["context"],
                                                            "paragraph_context": joined_paragraph_context,
                                                            "paragraph_token_positions": paragraph_token_positions,
                                                            "raw_question": question_and_answers['question'],
                                                            "question": question,
                                                            "a_start": [answer_start],
                                                            "a_end": [answer_end],
                                                            "a_extracted": [JOIN_TOKEN.join(
                                                                paragraph_context[answer_start:answer_end + 1])],
                                                            "a_gt": [possible_answer["text"]]}
                                else:
                                    examples[example_id]["a_start"].append(answer_start)
                                    examples[example_id]["a_end"].append(answer_end)
                                    examples[example_id]["a_extracted"].append(JOIN_TOKEN.join(
                                        paragraph_context[answer_start:answer_end + 1]))
                                    examples[example_id]["a_gt"].append(possible_answer["text"])
            if compact:
                examples = list(examples.values())
            # debug
            logging.info(f"# problems: {problems}")
            logging.info(f"Problems affect {problems / len(examples) / 100:.6f} % of dataset.")
            return examples

    @staticmethod
    def prepare_fields():
        WORD_field = data.Field(batch_first=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), lower=True)
        return [
            ('id', data.RawField()),
            ('topic_title', data.RawField()),
            ('document_token_positions', data.RawField()),
            ('raw_document_context', data.RawField()),
            ('document', WORD_field),
            ('document_char', data.RawField()),
            ('raw_document', data.RawField()),
            ('question', WORD_field),
            ('question_char', data.RawField()),
            ('raw_question', data.RawField()),
            ("a_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("a_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('ext_answer', data.RawField()),
            ('gt_answer', data.RawField())
        ]

    @staticmethod
    def prepare_fields_char(use_start_end_char_boundaries=False, include_lengths=True):
        WORD_field = data.Field(batch_first=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), lower=True,
                                include_lengths=include_lengths)
        if use_start_end_char_boundaries:
            CHAR_field = data.Field(batch_first=True, preprocessing=make_sure_min_length, tokenize=list,
                                    lower=True, init_token="<s>", eos_token="<e>")
        else:
            CHAR_field = data.Field(batch_first=True, preprocessing=make_sure_min_length, tokenize=list,
                                    lower=True)
        CHAR_nested_field = data.NestedField(CHAR_field, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN))
        return [
            ('id', data.RawField()),
            ('topic_title', data.RawField()),
            ('document_token_positions', data.RawField()),
            ('raw_document_context', data.RawField()),
            ('document', WORD_field),
            ('document_char', CHAR_nested_field),
            ('raw_document', data.RawField()),
            ('question', WORD_field),
            ('question_char', CHAR_nested_field),
            ('raw_question', data.RawField()),
            ("a_start", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ("a_end", data.Field(sequential=False, use_vocab=False, batch_first=True, is_target=True)),
            ('ext_answer', data.RawField()),
            ('gt_answer', data.RawField())
        ]

    @staticmethod
    def prepare_fields_char_noanswerprocessing(include_lengths=True, word_special_tokens=True):
        WORD_field = data.Field(batch_first=True, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN), lower=True,
                                include_lengths=include_lengths)
        if word_special_tokens:
            CHAR_field = data.Field(batch_first=True, preprocessing=make_sure_min_length, tokenize=list,
                                    lower=True, init_token="<s>", eos_token="<e>")
        else:
            CHAR_field = data.Field(batch_first=True, preprocessing=make_sure_min_length, tokenize=list, lower=True)
        CHAR_nested_field = data.NestedField(CHAR_field, tokenize=lambda s: str.split(s, sep=JOIN_TOKEN))
        return [
            ('id', data.RawField()),
            ('topic_title', data.RawField()),
            ('document_token_positions', data.RawField()),
            ('raw_document_context', data.RawField()),
            ('document', data.RawField()),
            ('document_char', data.RawField()),
            ('raw_document', data.RawField()),
            ('question', WORD_field),
            ('question_char', CHAR_nested_field),
            ('raw_question', data.RawField()),
            ("a_start", data.RawField()),
            ("a_end", data.RawField()),
            ('ext_answer', data.RawField()),
            ('gt_answer', data.RawField())
        ]
