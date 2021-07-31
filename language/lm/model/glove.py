import torch
from torch import nn
import torch.nn.functional as F


class Glove(nn.Module):

    def __init__(self, vocab, embedding_dim, c, alpha, **kwargs):
        """
        :param vocab: torchtext.vocab.Vocab.
        :param embedding_dim: int.
        :param c: float, c of weight.
        :param alpha: float, alpha of weight.
        """
        super().__init__()
        len_vocab = len(vocab)
        self.center_embedding = nn.Embedding(len_vocab, embedding_dim)
        self.center_bias = nn.Embedding(len_vocab, 1)
        self.context_embedding = nn.Embedding(len_vocab, embedding_dim)
        self.context_bias = nn.Embedding(len_vocab, 1)
        # for weight
        self.c = c
        self.alpha = alpha

    def forward(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(n), prediction.
        """
        center = data['center']
        context = data['context']
        center_embedding = self.center_embedding(center).unsqueeze(1)  # (n, 1, embedding_dim)
        context_embedding = self.context_embedding(context).unsqueeze(-1)  # (n, embedding_dim, 1)
        prediction = torch.bmm(center_embedding, context_embedding)  # (n, 1, 1)
        prediction = (prediction.squeeze()
                      + self.context_bias(context).squeeze()
                      + self.center_bias(center).squeeze())
        return prediction

    def forward_train(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(1), loss.
        """
        prediction = self(data)  # (n)
        frequency = data['frequency']

        # weight
        one = torch.tensor(1, dtype=torch.float).to(frequency.device)
        weight = torch.where(frequency < self.c, (frequency / self.c) ** self.alpha, one)

        loss = F.mse_loss(prediction, frequency.log(), reduction='none')
        loss = loss * weight
        loss = loss.sum() / weight.sum()
        return loss

    def get_embedding(self):
        return (self.center_embedding.weight.data.detach()
                + self.context_embedding.weight.data.detach())

