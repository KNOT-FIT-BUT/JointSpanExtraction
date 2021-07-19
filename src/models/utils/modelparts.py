import math
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.datasets.utility import find_sub_list
from src.scripts.tools.utility import get_device


class Embedder(torch.nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.scale_grad = config['scale_emb_grad_by_freq']
        self.embedding_dim = vocab.vectors.shape[1]
        self.embeddings = torch.nn.Embedding(len(vocab), self.embedding_dim, scale_grad_by_freq=self.scale_grad)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.embeddings.weight.requires_grad = False
        self.vocab = vocab

        logging.info(f"Optimize embeddings = {config['optimize_embeddings']}")
        logging.info(f"Scale grad by freq: {self.scale_grad}")
        logging.info(f"Vocabulary size = {len(vocab.vectors)}")

    def forward(self, input):
        return self.embeddings(input)


class CharEmbedder(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.embeddings = nn.Embedding(len(vocab), config["char_embedding_size"], padding_idx=1)
        self.embeddings.weight.data.uniform_(-0.001, 0.001)
        self.dropout = nn.Dropout(p=config["dropout_rate"])
        self.vocab = vocab
        self.char_conv = nn.Conv2d(1,  # input channels
                                   config["char_channel_size"],  # output channels
                                   (config["char_embedding_size"], config["char_channel_width"])  # kernel size
                                   )

    def forward(self, input):
        """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
        """

        batch_size = input.size(0)
        word_len = input.shape[-1]
        # (batch, seq_len, word_len, char_dim)
        x = self.dropout(self.embeddings(input))
        char_dim = x.shape[-1]
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, char_dim, word_len).unsqueeze(1)
        # (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv(x).squeeze(-2)
        # (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze(-1)
        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, x.shape[-1])
        return x


class HighwayNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = config["highway_layers"]
        dim = config["highway_dim1"] + config["highway_dim2"]
        for i in range(self.layers):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(nn.Linear(dim, dim),
                                  nn.ReLU()))
            gate = nn.Linear(dim, dim)

            # We should bias the highway layer to just carry its input forward when training starts.
            # We do that by setting the bias on gate affine transformation to be positive, because
            # that means `g` will be biased to be high, so we will carry the input forward.
            # The bias on `B(x)` is the second half of the bias vector in each Linear layer.
            gate.bias.data.fill_(1)

            setattr(self, f'highway_gate{i}',
                    nn.Sequential(gate,
                                  nn.Sigmoid()))

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        for i in range(self.layers):
            h = getattr(self, f'highway_linear{i}')(x)
            g = getattr(self, f'highway_gate{i}')(x)
            x = (1 - g) * h + g * x
        return x


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

    def get_output_dim(self):
        raise NotImplementedError("Objects need to implement this method!")


class RNN(Encoder):
    def __init__(self, config):
        super(RNN, self).__init__(config)
        self.rnn = None

    def forward(self, inp, lengths=None, padding_value=0., batch_first=True):
        """
        :param inp: Shape BATCH_SIZE x LEN x H_DIM
        """
        if lengths is None:
            outp = self.rnn(inp)[0]
        else:
            sequence_len = inp.shape[1]
            inp_packed = pack_padded_sequence(inp, lengths, batch_first=batch_first, enforce_sorted=False)
            outp_packed = self.rnn(inp_packed)[0]
            outp, output_lengths = pad_packed_sequence(outp_packed, batch_first=batch_first,
                                                       padding_value=padding_value, total_length=sequence_len)
        return outp

    def get_output_dim(self):
        return self.output_dim


class BiLSTM(RNN):
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config['RNN_nhidden']
        self.layers = config['RNN_layers']
        self.rnn = torch.nn.LSTM(
            config["RNN_input_dim"],
            self.hidden_size, self.layers,
            dropout=float(config['dropout_rate']),
            batch_first=True,
            bidirectional=True)
        self.output_dim = config['RNN_nhidden'] * 2


class LSTM(RNN):
    def __init__(self, config, init_hidden=None):
        super().__init__(config)
        self.hidden_size = config['RNN_nhidden']
        self.layers = config['RNN_layers']
        self.rnn = torch.nn.LSTM(
            config["RNN_input_dim"],
            self.hidden_size, self.layers,
            dropout=config['dropout_rate'],
            batch_first=True,
            bidirectional=False)
        self.output_dim = config['RNN_nhidden']


# @profile
def combine_surface_forms(valid_span_probabilities, batch_size, hacks, p_to_rerank, passage_length, score, pad_token=0):
    if score == "logprobs":
        # !!!!!sentinel is automatically assumed in this case!!!!
        # presoftmax class score = log(P_class) + K
        # save K, turn scores into probabilities
        K = torch.FloatTensor(
            np.nanmax((valid_span_probabilities - torch.log_softmax(valid_span_probabilities, -1)).cpu().numpy(), -1)) \
            .to(
            valid_span_probabilities.get_device() if valid_span_probabilities.get_device() >= 0 else torch.device(
                "cpu"))
        valid_span_probabilities = F.softmax(valid_span_probabilities, dim=-1)
        valid_span_probabilities = valid_span_probabilities.view(batch_size, passage_length, passage_length)
        valid_document_probabilities = valid_span_probabilities[:, 1:, 1:]
        valid_document_probabilities = valid_document_probabilities.reshape(batch_size, -1)
        passage_length -= 1
    else:
        valid_document_probabilities = valid_span_probabilities
    # Re-ranking top-N based on surface form
    sorted_scores, indices = torch.sort(valid_document_probabilities, dim=-1, descending=True)
    span_start_indices = indices // (passage_length)
    span_end_indices = indices % (passage_length)
    N = p_to_rerank  # top-N surface form reranking
    sorted_scores, span_start_indices, span_end_indices = sorted_scores[:, :N], \
                                                          span_start_indices[:, :N], \
                                                          span_end_indices[:, :N]
    if type(hacks["combine_surface_forms"][1]) == torch.Tensor:
        hacks["combine_surface_forms"] = hacks["combine_surface_forms"][0], \
                                         hacks["combine_surface_forms"][1].tolist()

    ### Casting to python floats may produce slightly different results, due to FP instability, e.g.:
    # 28.7.2020, changed to pytorch vectorized addition
    # ---------------------------------------------------------------------------------------
    # Python floats
    # 3.158890103804879e-05 + 2.225152506696304e-09
    # returns 3.1591126190555485e-05
    # ---------------------------------------------------------------------------------------
    # Pytorch vectorized addition of floats
    # (torch.Tensor([3.158890103804879e-05]) + torch.Tensor([2.225152506696304e-09]) ).item()
    # returns 3.159112748107873e-05

    # valid_document_probabilities_list = valid_document_probabilities.tolist()
    valid_document_probabilities_list = valid_document_probabilities

    for i in range(len(span_start_indices)):
        bool_arr_processed = [[False for _ in range(passage_length)] for _ in range(passage_length)]
        for a, e in zip(span_start_indices[i].tolist(), span_end_indices[i].tolist()):
            if bool_arr_processed[a][e]:
                continue

            # HERE assuming 0 in the pad token
            if hacks["combine_surface_forms"][1][i][a:e + 1] == [pad_token]:
                continue

            # OLD
            # processed.append((a, e))  # do not adjust value of other spans with this span
            bool_arr_processed[a][e] = True

            span_occurences = find_sub_list(hacks["combine_surface_forms"][1][i][a:e + 1],
                                            hacks["combine_surface_forms"][1][i])
            if len(span_occurences) > 1:
                for span in span_occurences:
                    if bool_arr_processed[span[0]][span[1]]:
                        continue
                    bool_arr_processed[span[0]][span[1]] = True

                    valid_document_probabilities_list[i][a * passage_length + e] += \
                        valid_document_probabilities_list[i][span[0] * passage_length + span[1]]

                    valid_document_probabilities_list[i][span[0] * passage_length + span[1]] = 0.

    # valid_document_probabilities = torch.FloatTensor(valid_document_probabilities_list)
    valid_document_probabilities = valid_document_probabilities_list
    if score == "logprobs":
        # turn back into pre-softmax scores
        valid_span_probabilities[:, 1:, 1:] = valid_document_probabilities.view(batch_size, passage_length,
                                                                                passage_length)
        valid_span_probabilities = valid_span_probabilities.view(batch_size, -1)
        valid_span_probabilities += K.unsqueeze(-1)
    return valid_span_probabilities


class SpanPredictionModule(nn.Module):
    def predict(self, batch):
        start_pred_logits, end_pred_logits = self(batch)
        start_pred, end_pred = torch.nn.functional.softmax(start_pred_logits, dim=1), torch.nn.functional.softmax(
            end_pred_logits, dim=1)
        return self.decode(start_pred, end_pred)

    @staticmethod
    def decode(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, has_sentinel=False, score="logprobs") -> \
            (torch.Tensor, torch.Tensor):
        """
        This method has been borrowed from AllenNLP
        :param span_start_logits:
        :param span_end_logits:
        :return:
        """
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)

        # if first token is sentinel, class, combinations (0,x) and (x,0); x!=0 are invalid
        # mask these
        if has_sentinel:
            span_log_probs[:, 1:, 0] = -math.inf
            span_log_probs[:, 0, 1:] = -math.inf

        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                              device=device)).log().unsqueeze(0)
        valid_span_log_probs = span_log_probs + span_log_mask

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
        if score == "probs":
            valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
        elif score == "logprobs":
            valid_span_scores = valid_span_log_probs
        else:
            raise NotImplemented(f"Unknown score type \"{score}\"")

        best_span_scores, best_spans = valid_span_scores.max(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        return best_span_scores, (span_start_indices, span_end_indices)

    @staticmethod
    def decode_wth_hacks(span_start_logits: torch.Tensor,
                         span_end_logits: torch.Tensor,
                         score="logprobs",
                         p_to_rerank=100,
                         has_sentinel=False,
                         hacks={
                             "max_answer_length": 30,
                             "combine_surface_forms": (False, None)
                         }) -> \
            (torch.Tensor, torch.Tensor):
        """
        This method has been borrowed from AllenNLP
        :param span_start_logits:
        :param span_end_logits:
        :return:
        """
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if "combine_surface_forms" not in hacks:
            hacks["combine_surface_forms"] = (False, None)
        if hacks["combine_surface_forms"][0]:
            assert hacks["combine_surface_forms"][1] is not None
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, passage_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)

        # if first token is sentinel, class, combinations (0,x) and (x,0); x!=0 are invalid
        # mask these
        if has_sentinel:
            span_log_probs[:, 1:, 0] = -math.inf
            span_log_probs[:, 0, 1:] = -math.inf
        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts.
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                              device=device)).log().unsqueeze(0)
        valid_span_log_probs = span_log_probs + span_log_mask

        spans_longer_than_maxlen_mask = torch.Tensor([[j - i + 1 > hacks["max_answer_length"]
                                                       for j in range(passage_length)] for i in range(passage_length)]) \
            .to(valid_span_log_probs.get_device() if valid_span_log_probs.get_device() >= 0 else torch.device("cpu"))
        valid_span_log_probs.masked_fill_(spans_longer_than_maxlen_mask.unsqueeze(0).bool(), -math.inf)

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)

        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
        if score == "probs":
            valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
        elif score == "logprobs":
            valid_span_scores = valid_span_log_probs
        else:
            raise NotImplemented(f"Unknown score type \"{score}\"")
        if hacks["combine_surface_forms"][0]:
            assert not (score == "probs" and has_sentinel), \
                "Not a supported variant - probability decoding + has_sentinel"
            pad_token_id = 0
            if len(hacks["combine_surface_forms"]) == 3:
                pad_token_id = hacks["combine_surface_forms"][-1]
            valid_span_scores = combine_surface_forms(valid_span_scores,
                                                      batch_size, hacks,
                                                      p_to_rerank, passage_length,
                                                      score, pad_token=pad_token_id)

        best_span_scores, best_spans = valid_span_scores.max(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        return best_span_scores, (span_start_indices, span_end_indices)

    @staticmethod
    def decode_topN_joint(valid_span_log_probs: torch.Tensor, N: int = 100) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        batch_size = valid_span_log_probs.shape[0]
        passage_length = valid_span_log_probs.shape[1]
        # Addition in log-domain = multiplication in real domain
        # This will create a matrix containing addition of each span_start_logit with span_end_logit
        # (batch_size, passage_length, passage_length)
        span_log_probs = valid_span_log_probs

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        # valid_span_log_probs is a vector [s_00,s_01,...,s_0n,s10,s11,...,s1n, ... , sn0,sn1,..., snn] of span scores
        # e.g. s_01 is a score of answer span from token 0 to token 1
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)  # see image above, part 2.

        # Turn all the log-probabilities into probabilities
        valid_span_probs = F.softmax(valid_span_log_probs, dim=-1)

        sorted_probs, indices = torch.sort(valid_span_probs, dim=-1, descending=True)
        # best_span_probs of shape batch_size now contains all probabilities for each best span in the batch
        # best_spans of shape batch_size now contains argmaxes of each answer from unrolled sequence valid_span_log_probs

        span_start_indices = indices // passage_length
        span_end_indices = indices % passage_length

        # return just N best
        return sorted_probs[:, :N], (span_start_indices[:, :N], span_end_indices[:, :N])

    @staticmethod
    def decode_topN_joint_wth_hacks(valid_span_log_probs: torch.Tensor, N: int = 100, score="probs", p_to_rerank=100,
                                    has_sentinel=False,
                                    hacks={
                                        "max_answer_length": 30,
                                        "combine_surface_forms": (False, None)
                                    }) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
                        This method has been borrowed from AllenNLP
                        :param valid_span_log_probs:
                        :return:
                """
        if "combine_surface_forms" not in hacks:
            hacks["combine_surface_forms"] = (False, None)
        if hacks["combine_surface_forms"][0]:
            assert hacks["combine_surface_forms"][1] is not None

        batch_size = valid_span_log_probs.shape[0]
        passage_length = valid_span_log_probs.shape[1]

        if has_sentinel:
            valid_span_log_probs[:, 1:, 0] = -math.inf
            valid_span_log_probs[:, 0, 1:] = -math.inf

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        spans_longer_than_maxlen_mask = torch.Tensor([[j - i + 1 > hacks["max_answer_length"]
                                                       for j in range(passage_length)] for i in range(passage_length)]) \
            .to(get_device(valid_span_log_probs))

        valid_span_log_probs.masked_fill_(spans_longer_than_maxlen_mask.unsqueeze(0).bool(), -math.inf)
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
        if score == "probs":
            valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
        elif score == "logprobs":
            valid_span_scores = valid_span_log_probs
        else:
            raise NotImplemented(f"Unknown score type \"{score}\"")

        if hacks["combine_surface_forms"][0]:
            assert not (score == "probs" and has_sentinel), \
                "Not a supported variant - proability decoding + has_sentinel"
            pad_token_id = 0
            if len(hacks["combine_surface_forms"]) == 3:
                pad_token_id = hacks["combine_surface_forms"][-1]
            valid_span_scores = combine_surface_forms(valid_span_scores,
                                                      batch_size, hacks,
                                                      p_to_rerank, passage_length,
                                                      score, pad_token=pad_token_id)

        sorted_probs, indices = torch.topk(valid_span_scores, k=N, dim=-1, largest=True)
        # best_span_probs of shape batch_size now contains topk probabilities for each best span in the batch
        # best_spans of shape batch_size now contains argmaxes of topk answers from unrolled sequence valid_span_log_probs

        span_start_indices = indices // passage_length
        span_end_indices = indices % passage_length

        return sorted_probs, (span_start_indices, span_end_indices)

    @staticmethod
    def decode_topN(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, N: int = 100) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        This method has been borrowed from AllenNLP
        :param span_start_logits: unnormalized start log probabilities
        :param span_end_logits: unnormalized end log probabilities
        :return:
        """
        # We call the inputs "logits" - they could either be unnormalized logits or normalized log
        # probabilities.  A log_softmax operation is a constant shifting of the entire logit
        # vector, so taking an argmax over either one gives the same result.
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, document_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device

        # span_start_logits.unsqueeze(2) has shape:
        # (batch_size, passage_length, 1)

        # span_end_logits.unsqueeze(1) has shape:
        # (batch_size, 1, passage_length)

        # Addition in log-domain = multiplication in real domain
        # This will create a matrix containing addition of each span_start_logit with span_end_logit
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)

        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts. We will mask these values out
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                              device=device)).log().unsqueeze(0)
        # The mask will look like this
        # 0000000
        # X000000
        # XX00000
        # XXX0000
        # XXXX000
        # XXXXX00
        # XXXXXX0
        # where X are -infinity
        valid_span_log_probs = span_log_probs + span_log_mask  # see image above, part 1.

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        # valid_span_log_probs is a vector [s_00,s_01,...,s_0n,s10,s11,...,s1n, ... , sn0,sn1,..., snn] of span scores
        # e.g. s_01 is a score of answer span from token 0 to token 1
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)  # see image above, part 2.

        # Turn all the log-probabilities into probabilities
        valid_span_probs = F.softmax(valid_span_log_probs, dim=-1)

        sorted_probs, indices = torch.sort(valid_span_probs, dim=-1, descending=True)
        # best_span_probs of shape batch_size now contains all probabilities for each best span in the batch
        # best_spans of shape batch_size now contains argmaxes of each answer from unrolled sequence valid_span_log_probs

        span_start_indices = indices // passage_length
        span_end_indices = indices % passage_length

        # return just N best
        return sorted_probs[:, :N], (span_start_indices[:, :N], span_end_indices[:, :N])

    @staticmethod
    def decode_topN_with_hacks(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, *args, **kwargs):
        if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
            raise ValueError("Input shapes must be (batch_size, document_length)")
        batch_size, passage_length = span_start_logits.size()
        device = span_start_logits.device

        # span_start_logits.unsqueeze(2) has shape:
        # (batch_size, passage_length, 1)

        # span_end_logits.unsqueeze(1) has shape:
        # (batch_size, 1, passage_length)

        # Addition in log-domain = multiplication in real domain
        # This will create a matrix containing addition of each span_start_logit with span_end_logit
        # (batch_size, passage_length, passage_length)
        span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)

        # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
        # the span ends before it starts. We will mask these values out
        span_log_mask = torch.triu(torch.ones((passage_length, passage_length),
                                              device=device)).log().unsqueeze(0)
        valid_span_log_probs = span_log_probs + span_log_mask
        return SpanPredictionModule.decode_topN_joint_wth_hacks(valid_span_log_probs, *args, **kwargs)

    @staticmethod
    def decode_joint(valid_span_log_probs: torch.Tensor, score="probs", has_sentinel=False) -> \
            (torch.Tensor, torch.Tensor):

        batch_size = valid_span_log_probs.shape[0]
        passage_length = valid_span_log_probs.shape[1]

        # if first token is sentinel, class, combinations (0,x) and (x,0); x!=0 are invalid
        # mask these
        if has_sentinel:
            valid_span_log_probs[:, 1:, 0] = -math.inf
            valid_span_log_probs[:, 0, 1:] = -math.inf

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
        if score == "probs":
            valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
        elif score == "logprobs":
            valid_span_scores = valid_span_log_probs
        else:
            raise NotImplemented(f"Unknown score type \"{score}\"")

        best_span_scores, best_spans = valid_span_scores.max(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        return best_span_scores, (span_start_indices, span_end_indices)

    @staticmethod
    def decode_joint_with_hacks(valid_span_log_probs: torch.Tensor, score="probs", p_to_rerank=100, has_sentinel=False,
                                hacks={
                                    "max_answer_length": 30,
                                    "combine_surface_forms": (False, None)
                                }) -> (torch.Tensor, torch.Tensor):
        """
                This method has been borrowed from AllenNLP
                :param valid_span_log_probs:
                :return:
        """
        if "combine_surface_forms" not in hacks:
            hacks["combine_surface_forms"] = (False, None)
        if hacks["combine_surface_forms"][0]:
            assert hacks["combine_surface_forms"][1] is not None

        batch_size = valid_span_log_probs.shape[0]
        passage_length = valid_span_log_probs.shape[1]

        # if first token is sentinel, class, combinations (0,x) and (x,0); x!=0 are invalid
        # mask these
        if has_sentinel:
            valid_span_log_probs[:, 1:, 0] = -math.inf
            valid_span_log_probs[:, 0, 1:] = -math.inf

        # Here we take the span matrix and flatten it, then find the best span using argmax.  We
        # can recover the start and end indices from this flattened list using simple modular
        # arithmetic.
        # (batch_size, passage_length * passage_length)
        spans_longer_than_maxlen_mask = torch.Tensor([[j - i + 1 > hacks["max_answer_length"]
                                                       for j in range(passage_length)] for i in range(passage_length)]) \
            .to(valid_span_log_probs.get_device() if valid_span_log_probs.get_device() >= 0 else torch.device("cpu"))
        valid_span_log_probs.masked_fill_(spans_longer_than_maxlen_mask.unsqueeze(0).bool(), -math.inf)
        valid_span_log_probs = valid_span_log_probs.view(batch_size, -1)
        if score == "probs":
            valid_span_scores = F.softmax(valid_span_log_probs, dim=-1)
        elif score == "logprobs":
            valid_span_scores = valid_span_log_probs
        else:
            raise NotImplemented(f"Unknown score type \"{score}\"")

        if hacks["combine_surface_forms"][0]:
            assert not (score == "probs" and has_sentinel), \
                "Not a supported variant - proability decoding + has_sentinel"

            pad_token_id = 0
            if len(hacks["combine_surface_forms"]) == 3:
                pad_token_id = hacks["combine_surface_forms"][-1]
            valid_span_scores = combine_surface_forms(valid_span_scores,
                                                      batch_size, hacks,
                                                      p_to_rerank, passage_length,
                                                      score, pad_token=pad_token_id)

        best_span_scores, best_spans = valid_span_scores.max(-1)
        span_start_indices = best_spans // passage_length
        span_end_indices = best_spans % passage_length

        return best_span_scores, (span_start_indices, span_end_indices)

    @staticmethod
    def decode_conditional(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor, top_k_start_positions,
                           beam_search_bestn, max_answer_length) -> \
            (torch.Tensor, torch.Tensor):
        best_starts = []
        best_ends = []
        span_scores = []
        max_n = []
        for i, batch in enumerate(span_end_logits):
            best_starts_for_b = torch.empty([beam_search_bestn, beam_search_bestn], dtype=torch.int)
            best_ends_for_b = torch.empty([beam_search_bestn, beam_search_bestn], dtype=torch.int)
            span_scores_for_b = torch.empty([beam_search_bestn, beam_search_bestn], dtype=torch.float)

            # iteration over top n start logits
            max_prob = float("-inf")
            max_n.append(0)
            for n, option in enumerate(span_end_logits[i]):
                end_logits_softmax = torch.nn.functional.softmax(span_end_logits[i][n], dim=-1)
                try:
                    start_logits_softmax = torch.nn.functional.softmax(span_start_logits[i], dim=-1)[
                        top_k_start_positions[i][n]]
                except IndexError as e:
                    print(e)
                    break
                total_prob = end_logits_softmax.max(-1)[0] + start_logits_softmax
                if total_prob > max_prob:
                    max_prob = total_prob
                    max_n[i] = n

                best_starts_for_b[n] = top_k_start_positions[i][n].repeat(beam_search_bestn)
                best_ends_for_b[n] = torch.topk(span_end_logits[i][n], beam_search_bestn).indices
                for j, be in enumerate(best_ends_for_b[n]):
                    span_scores_for_b[j] = torch.topk(end_logits_softmax, beam_search_bestn).values[
                                               j] + start_logits_softmax

            span_scores.append([float(s) for s in torch.flatten(span_scores_for_b)])
            best_starts.append([int(s) for s in torch.flatten(best_starts_for_b)])
            best_ends.append([int(e) for e in torch.flatten(best_ends_for_b)])

        start_indexes, end_indexes, best_span_scores, logprobs_S, logprobs_E0, logprobs_Emax = \
            best_starts, best_ends, span_scores, span_start_logits, span_end_logits[:, 0, :], \
            span_end_logits[torch.arange(span_end_logits.size(0)), max_n, :]

        best_scores_f = []
        start_indexes_f = []
        end_indexes_f = []
        for sib, eib, ssb in zip(start_indexes, end_indexes, best_span_scores):
            scores_l = []
            end_l = []
            start_l = []
            for si, ei, ss in zip(sib, eib, ssb):
                if ei - si <= max_answer_length and ei >= si:
                    scores_l.append(ss)
                    end_l.append(ei)
                    start_l.append(si)
            best_scores_f.append(scores_l)
            start_indexes_f.append(start_l)
            end_indexes_f.append(end_l)

        padded_S = torch.zeros(logprobs_E0.shape)
        padded_S[:logprobs_S.shape[0], :] = logprobs_S
        logprobs_S = padded_S

        return logprobs_S, logprobs_E0, logprobs_Emax, best_scores_f, start_indexes_f, end_indexes_f
