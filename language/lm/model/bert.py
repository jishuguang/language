import torch
from torch import nn
from torch.nn import functional as F


class Bert(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.bert_encoder = BertEncoder(**kwargs)
        self.mask_language_model = MaskedLanguageModel(**kwargs)
        self.next_sequence_predict = NextSentencePrediction(**kwargs)

    def forward(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(n, s, e), prediction.
        """
        embedding = self.bert_encoder(data)
        mlm_prediction = self.mask_language_model(embedding, data)
        nsp_prediction = self.next_sequence_predict(embedding, data)
        return mlm_prediction, nsp_prediction

    def forward_train(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(1), loss.
        """
        mlm_prediction, nsp_prediction = self(data)  # (n, k, v), (n, 2)

        # mlm loss
        predict_token = data['predict_token']  # (n, k)
        predict_mask = data['predict_mask']   # (n, k)
        mlm_loss = F.cross_entropy(mlm_prediction.permute(0, 2, 1), predict_token, reduction='none')  # (n, k)
        mlm_loss = (mlm_loss * predict_mask).sum() / (predict_mask.sum() + 1e-8)

        # nsp loss
        is_next = data['is_next']  # (n)
        nsp_loss = F.cross_entropy(nsp_prediction, is_next, reduction='mean')

        return mlm_loss + nsp_loss

    def get_embedding(self):
        return self.bert_encoder


class MaskedLanguageModel(nn.Module):

    def __init__(self, vocab, embedding_dim, num_dim_lm, **kwargs):
        """
        :param vocab: torchtext.vocab.Vocab.
        :param embedding_dim: int.
        :param num_dim_lm: int, dimension of hidden layers of masked language model.
        """
        super().__init__()
        len_vocab = len(vocab)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, num_dim_lm),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_dim_lm),
                                 nn.Linear(num_dim_lm, len_vocab))

    def forward(self, embedding, data):
        """
        :param embedding: Tensor, Shape(n, s, e), output of BertEncoder.
        :param data: dict, {key: Tensor}.
        :return: Tensor, Shape(n, k, v).
        """
        predict_index = data['predict_index']  # (n, k)
        batch_size = predict_index.shape[0]
        num_predict = predict_index.shape[1]
        batch_index = (torch.repeat_interleave(torch.arange(batch_size), num_predict)
                       .to(torch.int64)
                       .reshape(batch_size, -1))  # (n, k)
        embedding = embedding[batch_index, predict_index]
        prediction = self.mlp(embedding)  # (n, k, v)
        return prediction


class NextSentencePrediction(nn.Module):

    def __init__(self, embedding_dim, **kwargs):
        """
        :param embedding_dim: int.
        """
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, embedding, data):
        """
        :param embedding: Tensor, Shape(n, s, e), output of BertEncoder.
        :param data: dict, {key: Tensor}.
        :return: Tensor, Shape(n, 2).
        """
        prediction = self.linear(embedding[:, 0, :])
        return prediction


class BertEncoder(nn.Module):
    """Core of bert."""

    def __init__(self, vocab, embedding_dim, max_len,
                 num_layers, num_heads, feedforward_dim, dropout, **kwargs):
        """
        :param vocab: torchtext.vocab.Vocab.
        :param embedding_dim: int.
        :param max_len: max length of sequence.
        :param num_layers: number of encoder layers.
        :param num_heads: number of heads.
        :param feedforward_dim: dimension of feedforward layer.
        :param dropout: dropout of encoder layer.
        """
        super().__init__()

        # embedding
        len_vocab = len(vocab)
        self.token_embedding = nn.Embedding(len_vocab, embedding_dim)
        self.segment_embedding = nn.Embedding(2, embedding_dim)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                         nhead=num_heads,
                                                         dim_feedforward=feedforward_dim,
                                                         dropout=dropout,
                                                         batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(n, s, e), prediction.
        """
        tokens = data['tokens']  # (n, s)
        segments = data['segments']  # (n, s)
        pad_mask = data['pad_mask']  # (n, s)
        embedding = (self.token_embedding(tokens)  # (n, s, e)
                     + self.segment_embedding(segments)  # (n, s, e)
                     + self.pos_embedding.weight.unsqueeze(0)[:, :tokens.shape[1], :])  # (1, s, e)
        return self.encoder(embedding, src_key_padding_mask=pad_mask)
