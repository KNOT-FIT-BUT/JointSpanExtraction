import torch
import torch.nn.functional as F
from torch import nn

from src.models.utils.modelparts import BiLSTM,Embedder, HighwayNetwork, CharEmbedder, SpanPredictionModule


class BidAF(SpanPredictionModule):
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
        self.lin_E = nn.Linear(config["RNN_nhidden"] * 10, 1)
        self.dropout = nn.Dropout(p=config["dropout_rate"])

        self.att_weight_c = nn.Linear(config["RNN_nhidden"] * 2, 1, bias=False)
        self.att_weight_q = nn.Linear(config["RNN_nhidden"] * 2, 1, bias=False)
        self.att_weight_cq = nn.Linear(config["RNN_nhidden"] * 2, 1)

        config_modeling = config.copy()
        config_modeling["RNN_layers"] = 2
        config_modeling["RNN_input_dim"] = config["RNN_nhidden"] * 8
        self.modeling_layer = BiLSTM(config_modeling)

        config_output = config.copy()
        config_output["RNN_input_dim"] = config["RNN_nhidden"] * 14
        self.output_layer = BiLSTM(config_output)
        self.masking = True

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
        S = self.compute_similarity_matrix(d_enc, q_enc)
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

        return start_logits, end_logits

    def compute_similarity_matrix(self, c, q):
        """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
        """

        c_len, q_len = c.shape[1], q.shape[1]

        # "a" component similarity
        # (batch x c_len x 1) -> (batch x c_len x q_len)
        a_similarity = self.att_weight_c(c).expand(-1, -1, q_len)

        # "b" component similarity
        # (batch x q_len x 1) -> (batch x 1 x q_len) -> (batch x c_len x q_len)
        b_similarity = self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1)

        # Element wise similarity
        # A(B ⊙ w)
        # B = Q query_matrix (usually smaller), A = context_matrix
        element_wise_similarity = torch.bmm(c, (q * self.att_weight_cq.weight.unsqueeze(0))
                                            .transpose(1, 2)) + self.att_weight_cq.bias

        # Now add together to get total similarity
        total_similarity = a_similarity + b_similarity + element_wise_similarity

        return total_similarity

    ##############################
    # These are other variants of attention implementation I have tried before
    ##############################
    def compute_similarity_matrix_2(self, context_hiddens, question_hiddens, context_mask, question_mask,
                                    masking=False):
        """
        This is nicer implementation, but more memory expensive
        """
        batch_size = context_hiddens.shape[0]
        context_len = context_hiddens.shape[1]
        question_len = question_hiddens.shape[1]
        mask_size = (batch_size, context_len, question_len)
        mesh_size = (batch_size, context_len, question_len, context_hiddens.size(-1))
        context_mesh = context_hiddens.unsqueeze(dim=2).expand(mesh_size)
        question_mesh = question_hiddens.unsqueeze(dim=1).expand(mesh_size)
        attention_scores = self.similarity_linear(
            torch.cat([context_mesh, question_mesh, context_mesh * question_mesh], dim=-1)).squeeze(dim=-1)
        if masking:
            attention_scores.masked_fill_(
                context_mask.unsqueeze(dim=2).expand(mask_size) & question_mask.unsqueeze(dim=1).expand(mask_size),
                float('-inf'))
        return attention_scores

    def compute_similarity_matrix_3(self, c, q):
        """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
        This method implementation has been inspired with implementation of
        https://github.com/galsang/BiDAF-pytorch/blob/master/model/model.py
        """

        c_len, q_len = c.shape[1], q.shape[1]

        # "a" component similarity
        # (batch x c_len x 1) -> (batch x c_len x q_len)
        a_similarity = self.att_weight_c(c).expand(-1, -1, q_len)

        # "b" component similarity
        # (batch x q_len x 1) -> (batch x 1 x q_len) -> (batch x c_len x q_len)
        b_similarity = self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1)

        # Element wise similarity
        #####################################
        element_wise_similarity = []
        # Go through all the b vector representations
        for i in range(q_len):
            # (batch, 1, 2d)
            q_i = q.select(1, i).unsqueeze(1)

            # element-wise product them with all a vectors
            # (batch, c_len, 2d)
            elemw_sim = c * q_i

            # transform them to scalar representing element-wise component similarity score
            # (batch, c_len)
            element_wise_similarity_vector = self.att_weight_cq(elemw_sim).squeeze(-1)
            element_wise_similarity.append(element_wise_similarity_vector)

        # (batch, c_len, q_len)
        total_similarity = torch.stack(element_wise_similarity, dim=-1)

        # Now add together to get total similarity
        total_similarity += a_similarity + b_similarity

        return total_similarity

    ########################
    ########################

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
