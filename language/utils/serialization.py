import os

import torch
from language.utils.log import get_logger


logger = get_logger()


def save_model(model, path):
    logger.info(f'Save model to {path}.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, path)


def load_model(model, path, resume=False):
    # TODO: implement resume.
    logger.info(f'Loading model: {path}.')
    if not os.path.exists(path):
        raise FileExistsError(f'{path} does not exist.')

    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

