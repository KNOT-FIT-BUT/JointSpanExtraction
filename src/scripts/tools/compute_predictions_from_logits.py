import collections
import json
import logging

import torch
from tqdm import tqdm
from transformers.data.metrics.squad_metrics import get_final_text, _compute_softmax
from transformers.data.processors.squad import SquadResult

from src.datasets.utility import SquadJointResult
from src.models.utils.modelparts import SpanPredictionModule
from src.scripts.tools.find_best_threshold import find_best_threshold_squadutils_v2


def compute_predictions_from_logits(
        all_examples,
        all_features,
        all_results,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        verbose_logging,
        tokenizer,
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            if type(result) == SquadResult:
                best_span_scores, (span_start_indices, span_end_indices) = \
                    SpanPredictionModule.decode_wth_hacks(torch.FloatTensor(result.start_logits).unsqueeze(0),
                                                          torch.FloatTensor(result.end_logits).unsqueeze(0),
                                                          score="logprobs",
                                                          hacks={"max_answer_length": max_answer_length}
                                                          )
            else:
                assert type(result) == SquadJointResult
                best_span_scores, (span_start_indices, span_end_indices) = \
                    SpanPredictionModule.decode_joint_with_hacks(torch.FloatTensor(result.joint_logits).unsqueeze(0),
                                                                 score="logprobs",
                                                                 hacks={"max_answer_length": max_answer_length}
                                                                 )
            predictions.append(
                (best_span_scores[0].item(), (span_start_indices[0].item(), span_end_indices[0].item()),
                 feature_index))

        logit, (start_index, end_index), feature_index = sorted(predictions, key=lambda x: x[0], reverse=True)[0]
        feature = features[feature_index]
        tok_tokens = feature.tokens[start_index: (end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[start_index]
        orig_doc_end = feature.token_to_orig_map[end_index]
        orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

        tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
        all_predictions[example.qas_id] = final_text

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    return all_predictions


def get_predictions(
        all_examples,
        all_features,
        all_results,
        do_lower_case,
        output_prediction_file,
        verbose_logging,
        tokenizer,
        n_best_size,
        output_nbest_file,
        output_null_log_odds_file,
        threshold=None,
        version_2_with_negative=False,
        squad2_use_dev_tuned_noans_computation=False,
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    conditional = False

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result[0]] = result[1:]

    if version_2_with_negative:
        example_id_to_impossible = {example.qas_id: example.is_impossible for example in all_examples}

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "logit"]
    )
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    for (example_index, example) in tqdm(enumerate(all_examples), desc="Processing samples"):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        for (feature_index, feature) in enumerate(features):
            if len(unique_id_to_result[feature.unique_id]) == 6:
                feature_null_score, feature_score_E0, feature_score_E_maxn, best_span_scores, start_indexes, end_indexes \
                    = unique_id_to_result[feature.unique_id]
                conditional = True
            else:
                feature_null_score, best_span_scores, start_indexes, end_indexes \
                    = unique_id_to_result[feature.unique_id]
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    if conditional:
                        score_E0 = feature_score_E0
                        score_Emaxn = feature_score_E_maxn

                    min_null_feature_index = feature_index
            for start_index, end_index, logit in zip(start_indexes, end_indexes, best_span_scores):
                # We could hypothetically create invalid predictions, e.g., predict
                # that the start of the span is in the question. We throw out all
                # invalid predictions.
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_index,
                        end_index=end_index,
                        logit=logit,
                    )
                )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    logit=score_null,
                )
            )

        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, logit=pred.logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", logit=score_null))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["logit"] = entry.logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            # score_diff = score_null - best_non_null_entry.logit
            if squad2_use_dev_tuned_noans_computation:
                score_diff = 0.1708557903766632 * score_null + 0.08164593577384949 * score_E0 - 0.04595436900854111 * score_Emaxn - 0.6808629035949707 * best_non_null_entry.logit - 0.5985970497131348
            else:
                score_diff = score_null - best_non_null_entry.logit

            scores_diff_json[example.qas_id] = score_diff

            all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json

    if version_2_with_negative:
        if threshold is not None:
            logging.info(f"Threshold set to: {threshold}")
            null_score_diff_threshold = threshold
        else:
            logging.info("Seeking best no-answer threshold...")
            null_score_diff_threshold = find_best_threshold_squadutils_v2(example_id_to_impossible, scores_diff_json,
                                                                          all_predictions)
            logging.info(f"Found threshold: {null_score_diff_threshold:.3f}")

        for qas_id in all_predictions.keys():
            if scores_diff_json[qas_id] > null_score_diff_threshold:
                all_predictions[qas_id] = ""

    if output_prediction_file is not None:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file is not None:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        if output_null_log_odds_file is not None:
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions
