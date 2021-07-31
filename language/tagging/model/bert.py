import os

import torch
from torch import nn
from torch.nn import functional as F

from language.lm.model.bert import BertEncoder
from language.utils.log import get_logger


logger = get_logger()


class Bert(nn.Module):

    def __init__(self, tag_vocab, **kwargs):
        """
        :param tag_vocab: torchtext.vocab.Vocab.
        """
        super().__init__()
        self.bert_encoder = BertEncoder(**kwargs)

        # tagging
        tag_vocab_size = len(tag_vocab)
        embedding_dim = kwargs['embedding_dim']
        self.tag = nn.Linear(embedding_dim, tag_vocab_size)

    def forward(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, shape(n, s, v), prediction.
        """
        embedding = self.bert_encoder(data)  # (n, s, e)
        tag = self.tag(embedding)  # (n, s, v)
        return tag

    def forward_train(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, Shape(1), loss.
        """
        prediction = self(data)  # (n, s, v)
        tag = data['tags']  # (n, s)
        mask = data['tag_mask']  # (n, s)
        loss = F.cross_entropy(prediction.permute(0, 2, 1), tag, reduction='none')  # (n, s)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward_infer(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, Shape(n, s), prediction.
        """
        prediction = self(data)  # (n, s, v)
        predict_tag_index = torch.argmax(prediction, dim=2)  # (n, s)
        return predict_tag_index

    def load_bert_model(self, model_path):
        """
        :param model_path: path to bert pretrain model.
        :return: None.
        """
        logger.info(f'Loading model: {model_path}.')
        if not os.path.exists(model_path):
            raise FileExistsError(f'{model_path} does not exist.')

        self.bert_encoder = torch.load(model_path)
