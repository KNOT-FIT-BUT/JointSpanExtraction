import torch
import torch.nn.functional as F
from torch import nn

from src.models.utils.modelparts import BiLSTM, Embedder, HighwayNetwork, CharEmbedder, SpanPredictionModule


class BidAF_joint(SpanPredictionModule):
    """
    No character embeddings
    No highway network
    """

    def __init__(self, config, vocab, char_vocab):
        super().__init__()

        self.char_embedder = CharEmbedder(config, char_vocab)

        self.highway_network = HighwayNetwork(config)

        self.embedder = Embedder(vocab, config)

        self.pad_token = vocab.stoi["<pad>"]

        config_phrase = config.copy()
        config_phrase["RNN_input_dim"] = config["RNN_nhidden"] * 2
        self.encoder = BiLSTM(config_phrase)

        self.lin_S = nn.Linear(config["RNN_nhidden"] * 10, 1)

        self.dropout = nn.Dropout(p=config["dropout_rate"])

        self.att_weight_c = nn.Linear(config["RNN_nhidden"] * 2, 1, bias=False)
        self.att_weight_q = nn.Linear(config["RNN_nhidden"] * 2, 1, bias=False)
        self.att_weight_cq = nn.Linear(config["RNN_nhidden"] * 2, 1)

        if config.get("similarity_function") == "ff_mlp":
            hidden_size = config["RNN_nhidden"] * 10
            self.reader_logits = torch.nn.Linear(hidden_size, 1)
            self.projection_S = torch.nn.Linear(hidden_size, hidden_size)
            self.projection_E = torch.nn.Linear(hidden_size, hidden_size)

            self.layer_norm = torch.nn.LayerNorm(hidden_size)
        else:
            self.lin_J_S = nn.Linear(config["RNN_nhidden"] * 10, 1, bias=False)
            self.lin_J_E = nn.Linear(config["RNN_nhidden"] * 10, 1, bias=False)
            self.lin_J_SE = nn.Linear(config["RNN_nhidden"] * 10, 1)

        config_modeling = config.copy()
        config_modeling["RNN_layers"] = 2
        config_modeling["RNN_input_dim"] = config["RNN_nhidden"] * 8
        self.modeling_layer = BiLSTM(config_modeling)

        config_output = config.copy()
        config_output["RNN_input_dim"] = config["RNN_nhidden"] * 14
        self.output_layer = BiLSTM(config_output)
        self.masking = True
        self.config = config

    def get_device(self):
        return self.lin_S.weight.get_device() if self.lin_S.weight.get_device() >= 0 \
            else torch.device("cpu")

    def forward(self, document, question, document_char, question_char, mask_rnns=True):
        if mask_rnns:
            c_padding_mask = document[0] == self.pad_token
            q_padding_mask = question[0] == self.pad_token
        else:
            c_padding_mask = document == self.pad_token
            q_padding_mask = question == self.pad_token

        # 1. Character Embedding Layer
        q_emb_c, d_emb_c = self.char_embedder(question_char), self.char_embedder(document_char)

        # 2. Word Embedding Layer
        q_emb_w, d_emb_w = self.embedder(question[0] if mask_rnns else question), self.embedder(
            document[0] if mask_rnns else document)

        # Fuse the embeddings:
        q_emb, d_emb = self.highway_network(q_emb_w, q_emb_c), self.highway_network(d_emb_c, d_emb_w)

        # 3. Contextual embedding
        if mask_rnns:
            q_enc, d_enc = self.dropout(self.encoder(q_emb, lengths=question[1])), self.dropout(
                self.encoder(d_emb, lengths=document[1]))
        else:
            q_enc, d_enc = self.dropout(self.encoder(q_emb)), self.dropout(self.encoder(d_emb))

        # 4. Attention flow
        # (batch, c_len, q_len)
        S = self.compute_similarity_matrix(d_enc, q_enc, self.att_weight_c, self.att_weight_q, self.att_weight_cq)
        C2Q = self.co_attention(S, q_enc, q_padding_mask)
        Q2C = self.max_attention(S, d_enc, c_padding_mask)

        G = torch.cat((d_enc, C2Q, d_enc * C2Q, d_enc * Q2C), dim=-1)

        # (batch, c_len, 2d)
        if mask_rnns:
            M = self.dropout(self.modeling_layer(G, lengths=document[1]))
        else:
            M = self.dropout(self.modeling_layer(G))

        # span start representation
        start_rep = self.dropout(torch.cat((G, M), dim=-1))
        # (batch, c_len)
        start_logits_ = self.lin_S(start_rep).squeeze(-1)
        if self.masking:
            start_logits = start_logits_.masked_fill(c_padding_mask.bool(), float("-inf"))

        # start probs are still needed to compute end representations, when sticking to original formula
        start_probs = F.softmax(start_logits, dim=-1).unsqueeze(1)

        # (batch, 1 , 2d)
        st_summary = torch.bmm(start_probs, M)

        c_len = start_logits.shape[-1]
        TILL_START = st_summary.expand(st_summary.shape[0], c_len, st_summary.shape[-1])

        end_in_rep = torch.cat((G, M, TILL_START, M * TILL_START), dim=-1)
        if mask_rnns:
            M_2 = self.dropout(self.output_layer(end_in_rep, lengths=document[1]))
        else:
            M_2 = self.dropout(self.output_layer(end_in_rep))

        end_rep = self.dropout(torch.cat((G, M_2), dim=-1))

        # batch x start_logits (context_size) x end_logits_(context_size)
        if self.config["similarity_function"] == "simple_multi":
            joint_logits = torch.bmm(start_rep, end_rep.transpose(-1, -2))
        elif self.config["similarity_function"] == "ff_mlp":
            joint_logits = self.compute_similarity_matrix_mlp_orqa(self, start_rep, end_rep)
        else:
            joint_logits = self.compute_similarity_matrix(start_rep, end_rep,
                                                          self.lin_J_S, self.lin_J_E,
                                                          self.lin_J_SE,
                                                          att_type=self.config[
                                                              "similarity_function"])
        if self.masking:
            # sequences should never start/end from <s> or <pad> tokens
            joint_logits.masked_fill_(
                c_padding_mask.unsqueeze(dim=2).expand_as(joint_logits) | c_padding_mask.unsqueeze(
                    dim=1).expand_as(
                    joint_logits) | (torch.triu(torch.ones((joint_logits.shape[-1], joint_logits.shape[-1]),
                                                           device=self.get_device(),
                                                           )) < 0.1),
                float('-inf'))

        return joint_logits

    @staticmethod
    def compute_similarity_matrix_mlp_orqa(self, hidden_start, hidden_end):
        """
        :param sequence_output: batch x seq_len x hidden_dim
        """
        # https://github.com/google-research/language/blob/70c0328968d5ffa1201c6fdecde45bbc4fec19fc/language/orqa/models/orqa_model.py#L210
        # Score with an MLP to enable start/end interaction:
        # score(s, e) = w·σ(w_s·h_s + w_e·h_e)
        # where σ(x) = layer_norm(relu(x))
        # w_s, w_e have square shape: hidden x hidden
        # w = hidden x 1

        # project start/end representations
        hidden_start = self.projection_S(hidden_start)
        hidden_end = self.projection_E(hidden_end)

        bsz, seq_len, hidden = hidden_start.shape

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

    def max_attention(self, similarity_matrix, context_vectors, context_mask):
        """
        Compute max-attention w.r.expectation_embeddings_unilm. context
        :param similarity_matrix: (batch, c_len, q_len)
        :param context_vectors  (batch, c_len, hidden_size * 2)
        :return (batch, c_len, hidden_size * 2)
        """

        c_len = context_vectors.shape[1]

        # (batch, c_len)
        max_per_each_query = torch.max(similarity_matrix, dim=-1)[0]
        if self.masking:
            max_per_each_query.masked_fill_(context_mask.bool(), float("-inf"))
        max_similarities_across_queries = F.softmax(max_per_each_query, dim=-1)

        # (batch, 1, c_len)
        max_similarities_across_queries = max_similarities_across_queries.unsqueeze(1)
        # compare with each context word as dot product

        # (batch, 1, hidden_size * 2)
        s_max = torch.bmm(max_similarities_across_queries, context_vectors)

        S_max = s_max.expand(-1, c_len, -1)

        return S_max

    def co_attention(self, similarity_matrix, query_vectors, query_mask):
        """
        Compute co-attention w.r.expectation_embeddings_unilm. query
        :param similarity_matrix: (batch, c_len, q_len)
        :param query_vectors  (batch, q_len, hidden_size * 2)
        :param query_mask (batch, q_len)
        :return (batch, c_len, hidden_size * 2)
        """
        if self.masking:
            similarity_matrix = similarity_matrix.masked_fill(query_mask.unsqueeze(1).bool(), float("-inf"))
        s_soft = F.softmax(similarity_matrix, dim=-1)
        return torch.bmm(s_soft, query_vectors)
