import random

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import torchtext

from language.dataset.vocab import get_vocab
from language.utils.log import get_logger


logger = get_logger()


class BertDataset(Dataset):
    """
    Dataset for bert training.

    Reference: http://d2l.ai/chapter_natural-language-processing-pretraining/glove.html.
    """

    def __init__(self, data_path, vocab_path, min_freq, specials, ignore_case, max_len, **kwargs):
        """
        :param data_path: path to dataset.
        :param vocab_path: path to vocab file.
        :param min_freq: minimum frequency.
        :param specials: List[str], special symbols to add.
        :param ignore_case: bool, whether to ignore case.
        :param max_len: max length of sentence pair.
        """
        super().__init__()
        logger.info(f'Loading raw paragraphs from: {data_path}...')
        raw_paragraphs = self._get_raw_paragraphs(data_path)
        if ignore_case:
            raw_paragraphs = [[[token.lower() for token in line]
                               for line in paragraph]
                              for paragraph in raw_paragraphs]
        logger.info(f'Got {len(raw_paragraphs)} raw paragraphs.')

        raw_lines = [line for paragraph in raw_paragraphs for line in paragraph]
        self._vocab = get_vocab(vocab_path, raw_lines, min_freq, specials)
        logger.info(f'Vocab size: {len(self._vocab)}.')
        logger.info(f'Replace raw token with token index.')
        paragraphs = [[self._vocab.lookup_indices(line) for line in paragraph] for paragraph in raw_paragraphs]

        self._max_len = max_len
        sentence_pair_data = self._get_sentence_pair_data(paragraphs, self._vocab, max_len)
        self._data = self._setup_data(sentence_pair_data, self._vocab)
        logger.info(f'Dataset is ready, size: {len(self)}.')

    def _get_raw_paragraphs(self, data_path, **kwargs):
        """
        :param data_path: path to data.
        :return: List[List[List[token]]].
        """
        raise NotImplementedError

    @staticmethod
    def _get_sentence_pair_data(paragraphs, vocab, max_len):
        """
        Get sentence pair data.
        :param paragraphs: List[List[List[token_index]]]
        :param vocab: torchtext.vocab.Vocab.
        :param max_len: int, max length of sentence pair.
        :return: List[dict].
        """
        logger.info(f'Setup sentence pair data...')
        sentence_pair_data = list()
        paragraph_index_list = list(range(len(paragraphs)))
        for i, paragraph in tqdm(enumerate(paragraphs), total=len(paragraphs)):
            for j in range(len(paragraph) - 1):
                tokens_a = paragraph[j]

                # tokens_b
                is_next = random.random() < 0.5
                if is_next:
                    tokens_b = paragraph[j + 1]
                else:
                    chosen_paragraph_index = i
                    while chosen_paragraph_index == i:
                        chosen_paragraph_index = random.choice(paragraph_index_list)
                    tokens_b = random.choice(paragraphs[chosen_paragraph_index])

                # 1 '<cls>' token and 2 '<sep>' tokens
                redundant_len = len(tokens_a) + len(tokens_b) + 3 - max_len
                if redundant_len > 0:
                    left = int(len(tokens_a) / (len(tokens_a) + len(tokens_b)) * redundant_len)
                    right = redundant_len - left
                    tokens_a = tokens_a[left:]
                    tokens_b = tokens_b[:len(tokens_b) - right]

                # combine tokens
                tokens = [vocab['<cls>']] + tokens_a + [vocab['<sep>']] + tokens_b + [vocab['<sep>']]
                # 0 and 1 are marking segment A and B, respectively
                segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
                sentence_pair_data.append({
                    'tokens': tokens,
                    'segments': segments,
                    'is_next': is_next
                })
        logger.info(f'Got {len(sentence_pair_data)} sentence pair.')
        return sentence_pair_data

    @staticmethod
    def _setup_data(sentence_pair_data, vocab):
        """
        Setup data for bert training.
        :param sentence_pair_data: dict.
        :param vocab: torchtext.vocab.Vocab
        :return: List[dict].
        """
        logger.info(f'Setup data for bert training...')
        for data in tqdm(sentence_pair_data):
            tokens = data['tokens']
            invalid_tokens = set(vocab.lookup_indices(['<cls>', '<sep>', '<mask>', '<pad>']))
            candidate_index = [i for i, token in enumerate(tokens)
                               if token not in invalid_tokens]

            # 15% of random tokens are predicted in the masked language modeling task
            num_predict = max(1, int(len(tokens) * 0.15))
            index_predict = list()
            token_predict = list()
            random.shuffle(candidate_index)
            for index in candidate_index[:num_predict]:
                if random.random() < 0.8:
                    # 80% of the time: replace the word with the '<mask>' token
                    masked_token = vocab['<mask>']
                elif random.random() < 0.5:
                    # 10% of the time: keep the word unchanged
                    masked_token = tokens[index]
                else:
                    # 10% of the time: replace the word with a random word
                    while True:
                        masked_token = random.randint(0, len(vocab) - 1)
                        if masked_token not in invalid_tokens:
                            break
                    
                tokens[index] = masked_token
                index_predict.append(index)
                token_predict.append(masked_token)
                data['tokens'] = tokens
                data['predict_index'] = index_predict
                data['predict_token'] = token_predict
        return sentence_pair_data

    def __getitem__(self, index):
        data = self._data[index]
        valid_len = len(data['tokens'])
        pad_len = self._max_len - valid_len
        max_num_predict = int(self._max_len * 0.15)
        predict_pad_len = max_num_predict - len(data['predict_index'])
        return {
            'tokens': torch.tensor(data['tokens'] + [self._vocab['<pad>']] * pad_len, dtype=torch.int64),
            'segments': torch.tensor(data['segments'] + [0] * pad_len, dtype=torch.int),
            'pad_mask': torch.tensor([0] * valid_len + [1] * pad_len, dtype=torch.int),
            'is_next': torch.tensor(data['is_next'], dtype=torch.int64),
            'predict_index': torch.tensor(data['predict_index'] + [0] * predict_pad_len, dtype=torch.int64),
            'predict_token': torch.tensor(data['predict_token'] + [0] * predict_pad_len, dtype=torch.int64),
            'predict_mask': torch.tensor([1] * len(data['predict_index'] + [0] * predict_pad_len), dtype=torch.float)
        }

    def __len__(self):
        return len(self._data)

    def get_vocab(self):
        return self._vocab


class WikiText2(BertDataset):
    """WikiText2 dataset for bert embedding training."""

    def _get_raw_paragraphs(self, data_path, **kwargs):
        paragraphs = list()
        # use train split only
        for paragraph in tqdm(torchtext.datasets.WikiText2(data_path, split='train')):
            # treat . as the end of a sentence.
            lines = paragraph.strip().split(' . ')
            if len(lines) < 2:
                continue

            # tokenize word
            paragraph = list()
            for line in lines:
                tokens = line.strip().split()
                if tokens:
                    paragraph.append(tokens + ['.'])

            if len(paragraph) > 1:
                paragraphs.append(paragraph)
        return paragraphs


class WikiText2Test(WikiText2):
    """WikiText2 dataset for bert embedding training test."""

    def _get_raw_paragraphs(self, data_path, **kwargs):
        paragraphs = super()._get_raw_paragraphs(data_path)
        logger.info(f'Randomly select 1024 pieces of data for test.')
        random.shuffle(paragraphs)
        return paragraphs[:1024]


DATASETS = {
    'wikitext2_test': WikiText2Test,
    'wikitext2': WikiText2
}


def build_bert_dataset(**kwargs):
    name = kwargs['name']
    if name in DATASETS:
        logger.info(f'Building bert dataset {name}...')
        return DATASETS[name](**kwargs)
    else:
        raise ValueError(f'Invalid bert dataset name: {name}.')
