from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchtext

from language.dataset.vocab import get_vocab
from language.utils.log import get_logger


logger = get_logger()


class SequenceDataset(Dataset):
    """Sequence dataset for tagging."""

    def __init__(self, data_path, purpose, ignore_case,
                 vocab_path, min_freq, specials, tag_vocab_path,
                 max_len, **kwargs):
        """
        :param data_path: path to dataset.
        :param purpose: str, train/val/test.
        :param ignore_case: bool, whether to ignore case.
        :param vocab_path: path to vocab file.
        :param min_freq: minimum frequency.
        :param specials: List[str], special symbols to add.
        :param max_len: max length of sentence.
        """
        super().__init__()
        logger.info(f'Loading raw lines/tags from: {data_path}...')
        raw_lines, raw_tags = self._get_raw_data(data_path, purpose)
        if ignore_case:
            raw_lines = [[token.lower() for token in line] for line in raw_lines]
        logger.info(f'Got {len(raw_lines)} raw lines/tags.')

        self._vocab = get_vocab(vocab_path, raw_lines, min_freq, specials)
        logger.info(f'Vocab size: {len(self._vocab)}.')
        self._tag_vocab = get_vocab(tag_vocab_path, raw_tags, min_freq=1, specials=['<unk>'])
        logger.info(f'Tag vocab size: {len(self._tag_vocab)}')
        logger.info(f'Replace raw token/tag with token/tag index.')
        lines = [self._vocab.lookup_indices(line) for line in raw_lines]
        tags = [self._tag_vocab.lookup_indices(tag) for tag in raw_tags]

        self._data = self._setup_data(lines, tags, max_len)
        logger.info(f'Dataset is ready, size: {len(self)}.')

    def _get_raw_data(self, data_path, purpose, **kwargs):
        """
        :param data_path: path to data.
        :param purpose: str, train/val/test.
        :return: (List[List[token]], List[List[tag]]), (lines, tags).
        """
        raise NotImplementedError

    def _setup_data(self, lines, tags, max_len):
        """
        Setup data for sequence tagging.
        :param lines: List[List[token_index]].
        :param tags: List[List[tag_index]].
        :param max_len: max length of sentence.
        :return: dict.
        """
        filtered_lines = list()
        filtered_tags = list()
        tag_mask = list()
        pad_mask = list()
        segments = list()
        valid_lens = list()
        for i, line in enumerate(lines):
            if len(line) + 2 > max_len:
                continue
            valid_len = len(line)
            line = [self._vocab['<cls>']] + line + [self._vocab['<sep>']]
            pad_len = max_len - len(line)
            filtered_lines.append(line + [self._vocab['<pad>']] * pad_len)
            filtered_tags.append([0] + tags[i] + [0] * (pad_len + 1))
            segments.append([0] * (valid_len + 2) + [1] * pad_len)
            pad_mask.append([0] * (valid_len + 2) + [1] * pad_len)
            tag_mask.append([0] + [1] * valid_len + [0] * (pad_len + 1))
            valid_lens.append(valid_len)
        return {
            'tokens': torch.tensor(filtered_lines, dtype=torch.long),  # BertEncoder
            'tags': torch.tensor(filtered_tags, dtype=torch.long),  # Tagging
            'segments': torch.tensor(segments, dtype=torch.long),  # BertEncoder
            'pad_mask': torch.tensor(pad_mask, dtype=torch.int),  # BertEncoder
            'tag_mask': torch.tensor(tag_mask, dtype=torch.float),  # Tagging
            'valid_len': torch.tensor(valid_lens, dtype=torch.long)  # Tagging
        }

    def __getitem__(self, index):
        return {key: data[index] for key, data in self._data.items()}

    def __len__(self):
        return len(self._data['tokens'])

    def get_tag_vocab(self):
        return self._tag_vocab

    def get_vocab(self):
        return self._vocab


class Conll2000Chunking(SequenceDataset):
    """Conll dataset for tagging."""

    def _get_raw_data(self, data_path, purpose, **kwargs):
        lines = list()
        tags = list()
        if purpose == 'val':
            # here 'test' split is indeed 'val'
            purpose = 'test'
        for line, _, tag in tqdm(torchtext.datasets.CoNLL2000Chunking(data_path, split=purpose)):
            lines.append(line)
            tags.append(tag)
        return lines, tags


class Conll2000ChunkingTest(Conll2000Chunking):
    """Conll dataset for tagging test."""

    def _get_raw_data(self, data_path, purpose, **kwargs):
        lines, tags = super()._get_raw_data(data_path, purpose, **kwargs)
        logger.info(f'Select the first 1024 pieces of data for test.')
        return lines[:1024], tags[:1024]


DATASETS = {
    'conll_test': Conll2000ChunkingTest,
    'conll': Conll2000Chunking
}


def build_sequence_dataset(**kwargs):
    name = kwargs['name']
    if name in DATASETS:
        logger.info(f'Dataset purpose is set to {kwargs["purpose"]}...')
        return DATASETS[name](**kwargs)
    else:
        raise ValueError(f'Invalid dataset name: {name}.')
