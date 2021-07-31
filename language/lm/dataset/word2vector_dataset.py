from collections import Counter
import random

import torch
import torchtext
from torch.utils.data import Dataset
from tqdm import tqdm

from language.dataset.vocab import get_vocab
from language.utils.random import RandomChoices
from language.utils.log import get_logger


logger = get_logger()


class Word2VectorDataset(Dataset):
    """
    Dataset for Word2Vector training.

    Reference: http://d2l.ai/chapter_natural-language-processing-pretraining/word-embedding-dataset.html.
    """

    def __init__(self, data_path, vocab_path, min_freq=1, specials=None, ignore_case=True,
                 window=5, negative_amount=5, **kwargs):
        """
        :param data_path: path to dataset.
        :param vocab_path: path to vocab file.
        :param min_freq: minimum frequency.
        :param specials: List[str], special symbols to add.
        :param ignore_case: bool, whether to ignore case.
        :param window: int, context window.
        :param negative_amount: amount of negatives per context word.
        """
        logger.info(f'Loading raw lines from: {data_path}...')
        raw_lines = self._get_raw_lines(data_path, **kwargs)
        if ignore_case:
            raw_lines = [[token.lower() for token in line]for line in raw_lines]
        self._vocab = get_vocab(vocab_path, raw_lines, min_freq, specials)
        logger.info(f'Vocab size: {len(self._vocab)}.')

        logger.info(f'Replace raw token with token index.')
        lines = [self._vocab.lookup_indices(line) for line in raw_lines]
        logger.info(f'Counting token frequency...')
        counter = Counter([token for line in lines for token in line])

        centers, contexts = self._get_centers_contexts(lines, window)
        negatives = self._negative_sample(centers, contexts, negative_amount, counter)
        self._data = self._setup_data(centers, contexts, negatives)
        logger.info(f'Dataset is ready, size: {len(self)}.')

    def _get_raw_lines(self, data_path, **kwargs):
        """
        :param data_path: path to data.
        :return: List[List[token]].
        """
        raise NotImplementedError

    @staticmethod
    def _get_centers_contexts(lines, window):
        """
        :param lines: List[List[token_index]].
        :param window: int, context window.
        :return: (List[token_index], List[List[token_index]]), (centers, contexts).
        """
        centers = list()
        contexts = list()
        logger.info(f'Get centers and contexts from lines...')
        for line in tqdm(lines):
            if len(line) < 2:
                continue
            centers += line
            for i in range(len(line)):  # Context window centered at i
                window_size = random.randint(1, window)
                indices = list(range(max(0, i - window_size),
                                     min(len(line), i + 1 + window_size)))
                # Exclude the central word
                indices.remove(i)
                contexts.append([line[idx] for idx in indices])
        logger.info(f'Got {len(centers)} centers.')
        return centers, contexts

    @staticmethod
    def _negative_sample(centers, contexts, negative_amount, counter):
        """
        :param centers: List[token_index].
        :param contexts: List[List[token_index]].
        :param negative_amount: amount of negatives per context word.
        :param counter: Count, {word: frequency}.
        :return: List[List[token_index]], negatives.
        """
        indexes = list(counter.keys())
        sampling_weights = [count ** 0.75 for count in counter.values()]
        random_choices = RandomChoices(indexes, sampling_weights, int(0.1 * len(indexes)))
        negatives = list()
        logger.info(f'Get negative samples...')
        for center, context in tqdm(zip(centers, contexts), total=len(centers)):
            invalid_set = set(context + [center])
            negative = set()
            len_negative = int(len(context) * negative_amount)
            while len(negative) < len_negative:
                neg = random_choices()
                # Noise words cannot be context words
                if neg not in invalid_set:
                    negative.add(neg)
            negatives.append(list(negative))
        return negatives

    @staticmethod
    def _setup_data(centers, contexts, negatives):
        """
        Setup data for embedding training.
        TODO: consume too much memory.
        :param centers: List[token_index].
        :param contexts: List[List[token_index]].
        :param negatives: List[List[token_index]].
        :return: List[dict{key: Tensor}].
        """
        max_len = max(len(c) + len(n) for c, n in zip(contexts, negatives))
        data = list()
        logger.info(f'Setup dataset for training...')
        for center, context, negative in tqdm(zip(centers, contexts, negatives), total=len(centers)):
            # indicate which one is valid
            len_valid = len(context) + len(negative)
            mask = [1] * len_valid + [0] * (max_len - len_valid)
            # indicate which one is positive
            len_positive = len(context)
            label = [1] * len_positive + [0] * (max_len - len_positive)
            # context
            context = context + negative + [0] * (max_len - len_valid)
            data.append({
                'center': torch.tensor(center).to(torch.int64),
                'context': torch.tensor(context).to(torch.int64),
                'mask': torch.tensor(mask).to(torch.float),
                'label': torch.tensor(label).to(torch.float)
            })
        return data

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def get_vocab(self):
        return self._vocab


class WikiText2(Word2VectorDataset):
    """WikiText2 dataset for Word2Vector embedding training."""

    def _get_raw_lines(self, data_path, **kwargs):
        lines = list()
        # use train split only
        for line in tqdm(torchtext.datasets.WikiText2(data_path, split='train')):
            tokens = line.strip().split()
            if tokens:
                lines.append(tokens)
        return lines


class WikiText103(Word2VectorDataset):
    """WikiText103 dataset for Word2Vector embedding training."""

    def _get_raw_lines(self, data_path, **kwargs):
        lines = list()
        # use train split only
        for line in tqdm(torchtext.datasets.WikiText103(data_path, split='train')):
            tokens = line.strip().split()
            if tokens:
                lines.append(tokens)
        return lines


class Imdb(Word2VectorDataset):
    """Imdb dataset for Word2Vector embedding training."""

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


def build_word2vector_dataset(name, **kwargs):
    if name in DATASETS:
        return DATASETS[name](**kwargs)
    else:
        raise ValueError(f'Invalid word2vector dataset name {name}.')
