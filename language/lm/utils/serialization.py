import os

import torch

from language.utils.log import get_logger


logger = get_logger()


def save_embedding(embedding, path):
    """
    Save word embedding tensor.
    :param embedding: Tensor, embedding tensor.
    :param path: path to save.
    :return: None.
    """
    logger.info(f'Dumping word embedding to {path}.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(embedding, path)


def load_embedding(path):
    """
    Load word embedding tensor.
    :param path: path to embedding.
    :return: Tensor, word embedding.
    """
    logger.info(f'Loading word embedding from {path}.')
    if not os.path.exists(path):
        raise FileExistsError(f'{path} does not exist.')

    return torch.load(path)
