import math
import torch
from transformers import BertModel, PreTrainedModel
import logging

class TransformerForQuestionAnsweringConditional(torch.nn.Module):

    def __init__(self, config, sep_token_id, pad_token_id, keep_cls_level_output, model_clz=BertModel,
                 inverse_segment_mask=False, mask_cls=True):
        super().__init__()

        self.config = config
        self.transformer = model_clz.from_pretrained(config["model_type"], cache_dir=config["cache_dir"])

        hidden_size = self.transformer.config.hidden_size

        # TODO: remove redundant projections
        # Weights for computing independent probabilities
        self.qa_outputs = torch.nn.Linear(hidden_size, 2)

        self.albertE = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.albertEE = torch.nn.Linear(hidden_size, 1)
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        # whether the CLS-level output should be kept at all, or thrown away
        self.keep_cls_level_output = keep_cls_level_output
        self.lin_J_SE = torch.nn.Linear(hidden_size, 1)
        self.lin_J_S = torch.nn.Linear(hidden_size, 1, bias=False)
        self.lin_J_E = torch.nn.Linear(hidden_size, 1, bias=False)
        self.joint_S_linear = torch.nn.Linear(hidden_size, hidden_size)

        # whether to mask the logits at all
        self.masking = True

        # whether to mask the cls-level output logits
        self.mask_cls = mask_cls
        self.beam_search_topn = 10
        # whether to mask logits based on segment mask or inverse segment mask
        # (is question or context in first segment?)
        self.inverse_segment_mask = inverse_segment_mask

        self.bs = config["validation_batch_size"]
        self.new_end = torch.nn.Parameter(torch.zeros(self.bs, 384))

        if self.config.get("conditional_use_ln", False):
            self.layer_norm = torch.nn.LayerNorm(hidden_size)

        self.init_weights(model_clz)

    def init_weights(self, clz):
        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.transformer, module))

    def get_device(self):
        return self.qa_outputs.weight.get_device()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                a_starts=None):
        outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, head_mask=head_mask)
        offset = int(not self.keep_cls_level_output)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, offset:]
        if self.masking:
            segment_mask = ~(token_type_ids[:, offset:].bool()) if self.inverse_segment_mask else \
                token_type_ids[:, offset:].bool()
            context_mask = segment_mask | (input_ids[:, offset:] == self.sep_token_id)
            if not self.mask_cls or a_starts is not None: # squad2 or training
                context_mask[:, 0] = False

        logits = self.qa_outputs(sequence_output)
        start_logits, _ = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)

        # TODO: replace for-loops with parallelized version
        if a_starts is not None:  # training
            if self.masking:
                start_logits[context_mask] -= math.inf


            topk_start_positions = a_starts
            for i, b in enumerate(topk_start_positions):
                if i == 0:
                    start_features = torch.index_select(sequence_output[i], 0, b).unsqueeze(0)
                else:
                    start_features = torch.cat(
                        [start_features, torch.index_select(sequence_output[i], 0, b).unsqueeze(0)])
            start_features = start_features.repeat(1, sequence_output.shape[1], 1)
            os = torch.cat([sequence_output, start_features], -1)

            end_logits = self.tanh(self.albertE(os))
            if self.config.get("conditional_use_ln", False):
                end_logits = self.layer_norm(end_logits)

            end_logits = self.albertEE(end_logits).squeeze(-1)
            if self.masking:
                # masking of ends
                end_logits[context_mask] -= math.inf
            return start_logits, end_logits

        else:  # validation
            # masking of starts
            if self.masking:
                start_logits[context_mask] -= math.inf

            # batch x beam_search_topn
            topk_start_positions = torch.topk(start_logits, self.beam_search_topn, dim=1)[1]

            # For-loop implementation
            # for i, b in enumerate(topk_start_positions):  # over batch
            #     for j, pos in enumerate(topk_start_positions[i]):  # over each topk position
            #         for n, so in enumerate(sequence_output[i]):  # over each representation
            #             self.top_k_combined[i][j][n] = torch.cat([so, sequence_output[i][pos]], -1)

            bs, seq_len, dim = sequence_output.shape
            # batch * beam_search_topn x hidden_dim
            start_reps = sequence_output.gather(1, topk_start_positions.unsqueeze(2).expand(bs, self.beam_search_topn,
                                                                                            dim))
            top_k_combined = torch.cat((sequence_output.unsqueeze(1).expand(bs, self.beam_search_topn, seq_len, dim),
                                        start_reps.unsqueeze(-2).expand(bs, self.beam_search_topn, seq_len, dim)), -1)

            # check the implementation matches for_loop implementation
            # assert self.top_k_combined.allclose(top_k_combined)

            end_logits = self.tanh(self.albertE(top_k_combined))
            if self.config.get("conditional_use_ln", False):
                end_logits = self.layer_norm(end_logits)
            end_logits = self.albertEE(end_logits).squeeze(-1)

            if self.masking:
                # masking of ends
                end_logits[context_mask.unsqueeze(1).expand(end_logits.shape[0], self.beam_search_topn, -1)] -= math.inf

            return start_logits, end_logits, topk_start_positions
