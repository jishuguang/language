import torch
from torch import nn
import torch.nn.functional as F


class SkipGram(nn.Module):

    def __init__(self, vocab, embedding_dim, **kwargs):
        """
        :param vocab: torchtext.vocab.Vocab.
        :param embedding_dim: int.
        """
        super().__init__()
        len_vocab = len(vocab)
        self.center_embedding = nn.Embedding(len_vocab, embedding_dim)
        self.context_embedding = nn.Embedding(len_vocab, embedding_dim)

    def forward(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(n, context_dim), prediction logits.
        """
        center = data['center']
        context = data['context']
        center_embedding = self.center_embedding(center).unsqueeze(1)  # (n, 1, embedding_dim)
        context_embedding = self.context_embedding(context)  # (n, context_dim, embedding_dim)
        prediction = torch.bmm(center_embedding, context_embedding.permute(0, 2, 1))  # (n, 1, context_dim)
        return prediction.squeeze()

    def forward_train(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(1), loss.
        """
        logtis = self(data)  # (n, context_dim)
        mask = data['mask']
        label = data['label']
        loss = F.binary_cross_entropy_with_logits(logtis, label, weight=mask, reduction='none')
        loss = (loss.sum(dim=1) / mask.sum(dim=1)).mean()
        return loss

    def get_embedding(self):
        return self.center_embedding.weight.data.detach()
