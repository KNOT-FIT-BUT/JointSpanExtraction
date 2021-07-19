import torch
import torch.nn.functional as F
from torch import nn

from src.models.bidaf.bidaf_joint import BidAF_joint
from src.models.utils.modelparts import BiLSTM, Embedder, HighwayNetwork, CharEmbedder, SpanPredictionModule


class BidAF_compound(SpanPredictionModule):
    """
    No character embeddings
    No highway network
    """

    def __init__(self, config, vocab, char_vocab):
        super().__init__()

        self.char_embedder = CharEmbedder(config, char_vocab)
        self.embedder = Embedder(vocab, config)

        self.highway_network = HighwayNetwork(config)

        self.pad_token = vocab.stoi["<pad>"]

        config_phrase = config.copy()
        config_phrase["RNN_input_dim"] = config["RNN_nhidden"] * 2
        self.encoder = BiLSTM(config_phrase)

        self.lin_S = nn.Linear(config["RNN_nhidden"] * 10, 1)
        self.lin_E = nn.Linear(config["RNN_nhidden"] * 10, 1)

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
        S = BidAF_joint.compute_similarity_matrix(d_enc, q_enc, self.att_weight_c, self.att_weight_q,
                                                  self.att_weight_cq)
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
        else:
            start_logits = start_logits_

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

        end_logits_ = self.lin_E(end_rep).squeeze(-1)
        if self.masking:
            end_logits = end_logits_.masked_fill(c_padding_mask.bool(), float("-inf"))
        else:
            end_logits = end_logits_

        # batch x start_logits (context_size) x end_logits_(context_size)
        if self.config["similarity_function"] == "simple_multi":
            joint_logits = torch.bmm(start_rep, end_rep.transpose(-1, -2))
        elif self.config["similarity_function"] == "ff_mlp":
            joint_logits = BidAF_joint.compute_similarity_matrix_mlp_orqa(self, start_rep, end_rep)
        else:
            joint_logits = BidAF_joint.compute_similarity_matrix(start_rep, end_rep,
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

        return start_logits, end_logits, joint_logits

    def get_device(self):
        return self.lin_S.weight.get_device() if self.lin_S.weight.get_device() >= 0 \
            else torch.device("cpu")

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
