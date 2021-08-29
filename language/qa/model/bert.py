import os

import torch
from torch import nn
from torch.nn import functional as F

from language.lm.model.bert import BertEncoder
from language.utils.log import get_logger


logger = get_logger()


class Bert(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.bert_encoder = BertEncoder(**kwargs)

        # QA
        embedding_dim = kwargs['embedding_dim']
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: (Tensor, Tensor), shape(n, s), start_prediction, end_prediction.
        """
        embedding = self.bert_encoder(data)  # (n, s, e)
        prediction = self.linear(embedding)  # (n, s, 2)

        segment = data['segments'].unsqueeze(-1)  # (n, s, 1)
        pad_mask = data['pad_mask'].unsqueeze(-1)  # (n, s, 1)
        valid_mask = segment - pad_mask
        valid_mask[:, 0, :] = 1  # <cls> is valid

        offset = torch.zeros_like(valid_mask, device=valid_mask.device)
        offset[valid_mask == 0] = -1E6
        prediction = prediction * valid_mask + offset

        start_pred = prediction[:, :, 0]  # (n, s)
        end_pred = prediction[:, :, 1]  # (n, s)
        return start_pred, end_pred

    def forward_train(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, Shape(1), loss.
        """
        start_pred, end_pred = self(data)  # (n, s)
        start = data['start'][:, 0]  # (n)
        end = data['end'][:, 0]  # (n)

        loss = F.cross_entropy(start_pred, start) + F.cross_entropy(end_pred, end)
        return loss

    def forward_infer(self, data):
        """
        :param data: dict, {key: Tensor}.
        :return: Tensor, Shape(n, 2), prediction.
        """
        start_pred, end_pred = self(data)  # (n, s)
        no_answer_score = start_pred[:, 0] + end_pred[:, 0]  # <cls>

        start_pred = start_pred[:, 1:]
        end_pred = end_pred[:, 1:]
        start_max_values, start_max_indies = torch.cummax(start_pred, dim=1)
        answer_score = start_max_values + end_pred
        max_answer_score, end_index = torch.max(answer_score, dim=1)  # (n)
        start_index = start_max_indies[torch.arange(end_index.shape[0], device=end_index.device), end_index]

        return no_answer_score > max_answer_score, start_index + 1, end_index + 1

    def load_bert_model(self, model_path):
        """
        :param model_path: path to bert pretrain model.
        :return: None.
        """
        logger.info(f'Loading bert model: {model_path}.')
        if not os.path.exists(model_path):
            raise FileExistsError(f'{model_path} does not exist.')

        self.bert_encoder = torch.load(model_path)
