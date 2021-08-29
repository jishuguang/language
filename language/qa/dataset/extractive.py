import random

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchtext

from language.dataset.vocab import get_vocab
from language.utils.log import get_logger


logger = get_logger()


class ExtractiveDataset(Dataset):
    """Extractive question answer dataset."""

    def __init__(self, data_path, purpose, tokenizer,
                 vocab_path, min_freq, specials,
                 max_len, **kwargs):
        super().__init__()
        """
        :param data_path: path to dataset.
        :param purpose: str, train/val/test.
        :param tokenizer: str, torch tokenizer.
        :param vocab_path: path to vocab file.
        :param min_freq: minimum frequency.
        :param specials: List[str], special symbols to add.
        :param max_len: max length of sentence.
        """
        logger.info(f'Loading raw lines/tags from: {data_path}...')
        passage, question, start, end = self._get_raw_data(data_path, purpose, tokenizer)
        logger.info(f'Got {len(passage)} raw passages/questions.')

        self._vocab = get_vocab(vocab_path, passage + question, min_freq, specials)
        logger.info(f'Vocab size: {len(self._vocab)}.')

        logger.info(f'Replace raw token with token index.')
        passage = [self._vocab.lookup_indices(line) for line in passage]
        question = [self._vocab.lookup_indices(line) for line in question]

        self._data = self._setup_data(passage, question, start, end, max_len)
        logger.info(f'Dataset is ready, size: {len(self)}.')

    def _get_raw_data(self, data_path, purpose, tokenizer, **kwargs):
        """
        :param data_path: path to data.
        :param purpose: str, train/val/test.
        :param tokenizer: str, torch tokenizer.
        :return: (List[List[token]], List[List[token]]), List[List[int]], List[List[int]]),
                 (passage, question, start, end).
        """
        raise NotImplementedError

    def _setup_data(self, passage, question, start, end, max_len):
        """
        Setup data for QA.
        :param passage: List[List[token_index]].
        :param question: List[List[token_index]].
        :param start: List[List[int]].
        :param end: List[List[int]].
        :param max_len: max length of sentence.
        :return: dict.
        """
        # Initialization for BertEncoder
        token_list = list()
        segment_list = list()
        pad_mask_list = list()

        # Initialization for QA
        max_ans_count = max([len(s) for s in start])
        logger.info(f'Max answer count is {max_ans_count}.')

        logger.info(f'Setup dataset for QA:')
        for i in tqdm(range(len(passage))):
            # token_a & segment
            token_a = [self._vocab['<cls>']] + question[i] + [self._vocab['<sep>']]
            segment = [0] * len(token_a) + [1] * (max_len - len(token_a))

            # token_b & pad_mask
            psg = passage[i]
            max_passage_len = max_len - len(token_a) - 1
            min_start = min(start[i])
            max_end = max(end[i])
            if len(psg) <= max_passage_len:
                # whole passage is reserved
                pad_len = max_passage_len - len(psg)
                token_b_start = 0
                token_b = psg
            else:
                # random select part of passage
                pad_len = 0
                if min_start == -1:
                    token_b_start = random.randint(0, len(psg) - max_passage_len)
                else:
                    min_token_b_start = max(0, max_end - max_passage_len + 1)
                    max_token_b_start = min(len(psg) - max_passage_len, min_start)
                    if min_token_b_start > max_token_b_start:
                        start[i] = start[i][0:1]
                        end[i] = end[i][0:1]
                        min_token_b_start = max(0, end[i][0] - max_passage_len)
                        max_token_b_start = min(len(psg) - max_passage_len, start[i][0])
                    token_b_start = random.randint(min_token_b_start, max_token_b_start)
                token_b = psg[token_b_start: token_b_start + max_passage_len]
            pad_mask = [0] * (max_len - pad_len) + [1] * pad_len
            token_b += [self._vocab['<sep>']] + [self._vocab['<pad>']] * pad_len
            token = token_a + token_b

            # start & end
            for indices in (start, end):
                if min_start == -1:
                    indices[i] = [0]  # <cls>
                else:
                    indices[i] = [index - token_b_start + len(token_a) for index in indices[i]]
                indices[i] += (max_ans_count - len(indices[i])) * indices[i][-1:]
                assert len(indices[i]) == max_ans_count

            assert len(token) == max_len
            assert len(segment) == max_len
            assert len(pad_mask) == max_len
            token_list.append(token)
            segment_list.append(segment)
            pad_mask_list.append(pad_mask)

        return {
            # for BertEncoder
            'tokens': torch.tensor(token_list, dtype=torch.int64),
            'segments': torch.tensor(segment_list, dtype=torch.int),
            'pad_mask': torch.tensor(pad_mask_list, dtype=torch.int),
            # for QA
            'start': torch.tensor(start, dtype=torch.long),
            'end': torch.tensor(end, dtype=torch.long)
        }

    def __getitem__(self, index):
        return {key: data[index] for key, data in self._data.items()}

    def __len__(self):
        return self._data['tokens'].shape[0]


class SquadDataset(ExtractiveDataset):
    """Squad dataset."""

    def _dataset(self, data_path, purpose):
        raise NotImplementedError

    def _get_raw_data(self, data_path, purpose, tokenizer, **kwargs):
        passage_list = list()
        question_list = list()
        start_list = list()
        end_list = list()
        tokenizer = torchtext.data.utils.get_tokenizer(tokenizer)
        if purpose == 'val':
            # here 'dev' split is indeed 'val'
            purpose = 'dev'
        for passage, question, answer, answer_start in tqdm(self._dataset(data_path, purpose)):
            passage_list.append(tokenizer(passage))
            question_list.append(tokenizer(question))
            if answer_start[0] == -1:
                # no answer
                start_list.append([-1])
                end_list.append([-1])
            else:
                # answer
                starts = list()
                ends = list()
                for i, start_index in enumerate(answer_start):
                    start = len(tokenizer(passage[:start_index]))
                    end = start + len(tokenizer(answer[i])) - 1
                    starts.append(start)
                    ends.append(end)
                start_list.append(starts)
                end_list.append(ends)
        return passage_list, question_list, start_list, end_list

    def get_vocab(self):
        return self._vocab


class Squad1Dataset(SquadDataset):

    def _dataset(self, data_path, purpose):
        return torchtext.datasets.SQuAD1(data_path, split=purpose)


class Squad1DatasetTest(Squad1Dataset):

    def _dataset(self, data_path, purpose):
        total_dataset = super()._dataset(data_path, 'dev')
        return [next(total_dataset) for _ in range(1000)]


class Squad2Dataset(SquadDataset):

    def _dataset(self, data_path, purpose):
        return torchtext.datasets.SQuAD2(data_path, split=purpose)


class Squad2DatasetTest(Squad2Dataset):

    def _dataset(self, data_path, purpose):
        total_dataset = super()._dataset(data_path, 'dev')
        return [next(total_dataset) for _ in range(1000)]


DATASETS = {
    'squad1': Squad1Dataset,
    'squad1_test': Squad1DatasetTest,
    'squad2': Squad2Dataset,
    'squad2_test': Squad2DatasetTest
}


def build_extractive_dataset(**kwargs):
    name = kwargs['name']
    if name in DATASETS:
        logger.info(f'Dataset purpose is set to {kwargs["purpose"]}...')
        return DATASETS[name](**kwargs)
    else:
        raise ValueError(f'Invalid dataset name: {name}.')
