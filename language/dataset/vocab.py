import os

import torch
import torchtext

from language.utils.log import get_logger


logger = get_logger()


def save_vocab(vocab, path):
    """
    Save torchtext.vocab.Vocab.
    :param vocab: torchtext.vocab.Vocab.
    :param path: path to save.
    :return: None.
    """
    logger.info(f'Dumping vocab to {path}.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(vocab, path)


def load_vocab(path):
    """
    Load torchtext.vocab.Vocab.
    :param path: path to vocab text file.
    :return: torch.text.Vocab.
    """
    logger.info(f'Loading vocab from {path}.')
    if not os.path.exists(path):
        raise FileExistsError(f'{path} does not exist.')

    return torch.load(path)


def get_vocab(vocab_path, raw_lines, min_freq, specials):
    """
    :param vocab_path: path to vocab text file.
    :param raw_lines: List[List[token]].
    :param min_freq: minimum frequency.
    :param specials: Special symbols to add.
    :return: torchtext.vocab.Vocab.
    """
    if os.path.exists(vocab_path):
        logger.info(f'Vocab path exists, just load.')
        voc = load_vocab(vocab_path)
    else:
        logger.info(f'Vocab path does not exist, build vocab from raw lines.')
        voc = torchtext.vocab.build_vocab_from_iterator(raw_lines, min_freq, specials, special_first=True)
        # <unk> should be contained in the vocabulary
        voc.set_default_index(voc['<unk>'])
        save_vocab(voc, vocab_path)
    return voc
