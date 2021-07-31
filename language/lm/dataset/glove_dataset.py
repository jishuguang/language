from collections import defaultdict
import math

import torch
import torchtext
from torch.utils.data import Dataset
from tqdm import tqdm

from language.dataset.vocab import get_vocab
from language.utils.log import get_logger


logger = get_logger()


class GloveDataset(Dataset):
    """
    Dataset for glove training.

    Reference: http://d2l.ai/chapter_natural-language-processing-pretraining/glove.html.
    """

    def __init__(self, data_path, vocab_path, min_freq=1, specials=None, ignore_case=True, window=5, **kwargs):
        """
        :param data_path: path to dataset.
        :param vocab_path: path to vocab file.
        :param min_freq: minimum frequency.
        :param specials: List[str], special symbols to add.
        :param ignore_case: bool, whether to ignore case.
        :param window: int, context window.
        """
        logger.info(f'Loading raw lines from: {data_path}...')
        raw_lines = self._get_raw_lines(data_path, **kwargs)
        if ignore_case:
            raw_lines = [[token.lower() for token in line] for line in raw_lines]

        self._vocab = get_vocab(vocab_path, raw_lines, min_freq, specials)
        logger.info(f'Vocab size: {len(self._vocab)}.')
        logger.info(f'Replace raw token with token index.')
        lines = [self._vocab.lookup_indices(line) for line in raw_lines]

        frequency_dict = self._get_frequency_dict(lines, window)
        self._data = self._setup_data(frequency_dict)
        logger.info(f'Dataset is ready, size: {len(self)}.')

    def _get_raw_lines(self, data_path, **kwargs):
        """
        :param data_path: path to data.
        :return: List[List[token]].
        """
        raise NotImplementedError

    @staticmethod
    def _get_frequency_dict(lines, window):
        """
        :param lines: List[List[token_index]].
        :param window: int, context window.
        :return: dict{center,context: frequency}
        """
        frequency_dict = defaultdict(float)
        logger.info(f'Get frequency dict from lines...')
        for line in tqdm(lines):
            if len(line) < 2:
                continue
            for i in range(len(line)):  # Context window centered at i
                indices = list(range(max(0, i - window),
                                     min(len(line), i + 1 + window)))
                # Exclude the central word
                indices.remove(i)
                for idx in indices:
                    frequency_dict[line[i], line[idx]] += 1 / math.fabs(idx - i)
        logger.info(f'Got {len(frequency_dict)} pairs.')
        return frequency_dict

    @staticmethod
    def _setup_data(frequency_dict):
        """
        Setup data for embedding training.
        :param frequency_dict: dict{center,context: frequency}.
        :return: List[dict{key: Tensor}].
        """
        data = list()
        logger.info(f'Setup data for embedding training...')
        for (center, context), frequency in tqdm(frequency_dict.items()):
            data.append({
                'center': torch.tensor(center, dtype=torch.int64),
                'context': torch.tensor(context, dtype=torch.int64),
                'frequency': torch.tensor(frequency, dtype=torch.float)
            })
        return data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def get_vocab(self):
        return self._vocab


class WikiText2(GloveDataset):
    """WikiText2 dataset for glove embedding training."""

    def _get_raw_lines(self, data_path, **kwargs):
        lines = list()
        # use train split only
        for line in tqdm(torchtext.datasets.WikiText2(data_path, split='train')):
            tokens = line.strip().split()
            if tokens:
                lines.append(tokens)
        return lines


class WikiText103(GloveDataset):
    """WikiText103 dataset for glove embedding training."""

    def _get_raw_lines(self, data_path, **kwargs):
        lines = list()
        # use train split only
        for line in tqdm(torchtext.datasets.WikiText103(data_path, split='train')):
            tokens = line.strip().split()
            if tokens:
                lines.append(tokens)
        return lines


class Imdb(GloveDataset):
    """Imdb dataset for glove embedding training."""

    def _get_raw_lines(self, data_path, **kwargs):
        lines = list()
        # use train split only
        for _, line in tqdm(torchtext.datasets.IMDB(data_path, split='train')):
            # TODO: split punctuation
            tokens = line.strip().split()
            if tokens:
                lines.append(tokens)
        return lines


DATASETS = {
    'wikitext2': WikiText2,
    'wikitext103': WikiText103,
    'imdb': Imdb
}


def build_glove_dataset(name, **kwargs):
    if name in DATASETS:
        return DATASETS[name](**kwargs)
    else:
        raise ValueError(f'Invalid glove dataset name {name}.')
