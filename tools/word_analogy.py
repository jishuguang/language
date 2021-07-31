import argparse

import torch

from language.dataset.vocab import load_vocab
from language.lm.utils.serialization import load_embedding
from language.utils.log import get_logger


logger = get_logger()


class TestEmbedding:

    def __init__(self, vocab, embedding):
        """
        :param vocab: torchtext.vocab.Vocab.
        :param embedding: Tensor.
        """
        self._vocab = vocab
        self._embedding = embedding

    def _get_similarity(self, index):
        """
        :param index: int, word index.
        :return: Tensor, shape(n).
        """
        word_embedding = self._embedding[index]
        cos_similarity = self._embedding @ word_embedding.T \
            / torch.sqrt(torch.sum(self._embedding ** 2, dim=1) * torch.sum(word_embedding ** 2) + 1e-9)
        return cos_similarity

    def test(self, number):
        """Input a word, print similar words."""
        while True:
            word = input('Please input a word (input -1 to exit): ')
            if word == '-1':
                break
            index = self._vocab[word]
            print(f'\"{word}\" index: {index}.')
            similarity = self._get_similarity(index)
            values, indices = torch.topk(similarity, number + 1)
            # exclude itself
            for value, index in zip(values[1:], indices[1:]):
                print(f'index: {index:7} | word: {self._vocab.lookup_token(index):14} | similarity: {value:.4f}.')


def main():
    parser = argparse.ArgumentParser(description='Word analogy based on language model.')
    parser.add_argument('--vocab', required=True, type=str, help='Path to vocab.')
    parser.add_argument('--embedding', required=True, type=str, help='Path to embedding.')
    parser.add_argument('--number', required=False, type=int, default=4, help='How many similar words to print.')
    args = parser.parse_args()

    vocab = load_vocab(args.vocab)
    embedding = load_embedding(args.embedding)

    TestEmbedding(vocab, embedding).test(args.number)


if __name__ == '__main__':
    main()
