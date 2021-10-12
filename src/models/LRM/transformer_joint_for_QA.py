import torch
from transformers import BertModel, PreTrainedModel


class TransformerForQuestionAnsweringJoint(torch.nn.Module):
    """
    In this answer, no-answer is predicted as a position 0,0
    """

    def init_weights(self, clz):
        for ch in self.children():
            if issubclass(ch.__class__, torch.nn.Module) and not issubclass(ch.__class__, PreTrainedModel):
                ch.apply(lambda module: clz._init_weights(self.transformer, module))

    def __init__(self, config, sep_token_id, pad_token_id, keep_cls_level_output, model_clz=BertModel,
                 inverse_segment_mask=False,
                 mask_cls=True):
        super().__init__()

        self.transformer = model_clz.from_pretrained(config["model_type"], cache_dir=config["cache_dir"])
        hidden_size = self.transformer.config.hidden_size

        if config["similarity_function"] == "ff_mlp":
            self.projection = torch.nn.Linear(hidden_size, hidden_size * 2)
            self.reader_logits = torch.nn.Linear(hidden_size, 1)
            self.layer_norm = torch.nn.LayerNorm(hidden_size)
        elif config["similarity_function"] in ["multi", "add", "multi_add"]:
            self.lin_J_SE = torch.nn.Linear(hidden_size, 1)
            self.lin_J_S = torch.nn.Linear(hidden_size, 1, bias=False)
            self.lin_J_E = torch.nn.Linear(hidden_size, 1, bias=False)
            self.joint_S_linear = torch.nn.Linear(hidden_size, hidden_size)
        elif config["similarity_function"] == "simple_multi":
            self.joint_S_linear = torch.nn.Linear(hidden_size, hidden_size)
        else:
            raise ValueError(f"Unknown attention type {config['similarity_function']}")

        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id

        # whether the CLS-level output should be kept at all, or thrown away
        self.keep_cls_level_output = keep_cls_level_output

        # whether to mask the joint logits at all
        self.masking = True

        # whether to mask the cls-level output logits
        self.mask_cls = mask_cls

        # whether to mask logits based on segment mask or inverse segment mask
        # (is question or context in first segment?)
        self.inverse_segment_mask = inverse_segment_mask

        self.init_weights(model_clz)

    @staticmethod
    def compute_similarity_matrix(c, q, att_weight_c, att_weight_q, att_weight_cq, att_type="multi_add"):
        """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
        This implementation should be both, faster and more memory efficient than compute_similarity_matrix_3 implementation
        """

        c_len, q_len = c.shape[1], q.shape[1]
        a_similarity = b_similarity = element_wise_similarity = 0.
        if att_type == "add" or att_type == "multi_add":
            # "a" component similarity
            # (batch x c_len x 1) -> (batch x c_len x q_len)
            a_similarity = att_weight_c(c).expand(-1, -1, q_len)

            # "b" component similarity
            # (batch x q_len x 1) -> (batch x 1 x q_len) -> (batch x c_len x q_len)
            b_similarity = att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1)

        if att_type == "multi" or att_type == "multi_add":
            # Element wise similarity
            # A(B ⊙ w)
            # B = Q query_matrix (usually smaller), A = context_matrix
            element_wise_similarity = torch.bmm(c, (q * att_weight_cq.weight.unsqueeze(0))
                                                .transpose(1, 2)) + att_weight_cq.bias

        # Now add together to get total similarity
        total_similarity = a_similarity + b_similarity + element_wise_similarity

        return total_similarity

    def compute_similarity_matrix_mlp_orqa(self, sequence_output):
        """
        :param sequence_output: batch x seq_len x hidden_dim
        """
        # https://github.com/google-research/language/blob/70c0328968d5ffa1201c6fdecde45bbc4fec19fc/language/orqa/models/orqa_model.py#L210
        # Score with an MLP to enable start/end interaction:
        # score(s, e) = w·σ(w_s·h_s + w_e·h_e)
        # where σ(x) = layer_norm(relu(x))
        # w_s, w_e have square shape: hidden x hidden
        # w = hidden x 1

        bsz, seq_len, hidden = sequence_output.shape

        ############################
        # create hidden representations
        ############################
        # both are of shape batch x len x hidden
        hidden_start, hidden_end = self.projection(sequence_output).split(hidden, dim=-1)

        ############################
        # outer summation
        ############################

        # (batch x seq_len x hidden) -> (batch x seq_len x 1 x hidden)
        hidden_start_ = hidden_start.unsqueeze(-2)

        # (batch x seq_len x hidden) -> (batch x 1 x seq_len x hidden)
        hidden_end_ = hidden_end.unsqueeze(-3)
        # batch x seq_len x seq_len x hidden
        outer_sum = hidden_start_ + hidden_end_

        # Tests
        # with torch.no_grad():
        #     assert (hidden_start[0, 1] + hidden_end[0, 5]).allclose(outer_sum[0, 1, 5])
        #     assert not (hidden_start[0, 1] + hidden_end[0, 5]).allclose(outer_sum[0, 5, 1])
        #     assert (hidden_start[0, 5] + hidden_end[0, 1]).allclose(outer_sum[0, 5, 1])
        #     assert not (hidden_start[0, 5] + hidden_end[0, 1]).allclose(outer_sum[0, 1, 5])

        ############################
        # projection and activation
        ############################
        # score(s, e) = w·σ(w_s·h_s + w_e·h_e)
        # where σ(x) = layer_norm(relu(x))
        # (batch x seq_len x seq_len x hidden) -> (batch x seq_len^2 x hidden)
        outer_sum_linearized = outer_sum.view(bsz, -1, hidden)
        activated_linearized = self.layer_norm(torch.relu(outer_sum_linearized))

        logits = self.reader_logits(activated_linearized).view(bsz, seq_len, seq_len)

        # Tests
        # σ = lambda x: self.reader_logits(self.layer_norm(torch.relu(x))).squeeze(-1)
        # with torch.no_grad():
        #     assert σ(hidden_start[0, 1] + hidden_end[0, 5]).allclose(logits[0, 1, 5])
        #     assert not σ(hidden_start[0, 1] + hidden_end[0, 5]).allclose(logits[0, 5, 1])
        #     assert σ(hidden_start[0, 5] + hidden_end[0, 1]).allclose(logits[0, 5, 1])
        #     assert not σ(hidden_start[0, 5] + hidden_end[0, 1]).allclose(logits[0, 1, 5])

        return logits

    def get_device(self):
        return self.transformer.device

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                   attention_mask=attention_mask, head_mask=head_mask)

        offset = int(not self.keep_cls_level_output)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, offset:]

        joint_logits = self.compute_joint_scores(self, sequence_output)
        if self.masking:
            context_mask = (~token_type_ids[:, offset:].bool() \
                                if self.inverse_segment_mask else token_type_ids[:, offset:].bool()) \
                           | (input_ids[:, offset:] == self.sep_token_id) \
                           | (input_ids[:, offset:] == self.pad_token_id)
            if not self.mask_cls:
                context_mask[:, 0] = False
            # sequences should never start/end with <s> or <pad> tokens
            joint_logits.masked_fill_(
                context_mask.unsqueeze(dim=2).expand_as(joint_logits) |
                context_mask.unsqueeze(dim=1).expand_as(joint_logits) |
                (torch.triu(
                    torch.ones(
                        (joint_logits.shape[-1], joint_logits.shape[-1]),
                        device=self.get_device())
                ) < 0.1),
                float('-inf'))
            if self.keep_cls_level_output:
                joint_logits[:, 1:, 0] = float('-inf')
                joint_logits[:, 0, 1:] = float('-inf')

        return joint_logits

    @staticmethod
    def compute_joint_scores(self, sequence_output):
        if self.config.get("similarity_function", "simple_multi") == "ff_mlp":
            joint_logits = self.compute_similarity_matrix_mlp_orqa(sequence_output)
        else:
            start_rep = self.joint_S_linear(sequence_output)
            end_rep = sequence_output
            if self.config.get("similarity_function", "simple_multi") == "simple_multi":
                joint_logits = torch.bmm(start_rep, end_rep.transpose(-1, -2))
            elif self.config["similarity_function"] in ["add", "multi", "multi_add"]:
                joint_logits = self.compute_similarity_matrix(start_rep, end_rep, self.lin_J_S,
                                                              self.lin_J_E, self.lin_J_SE,
                                                              att_type=self.config["similarity_function"])
            else:
                raise ValueError(f"Unknown attention type: {self.config['similarity_function']}")
        return joint_logits
