from .word2vector_dataset import build_word2vector_dataset
from .glove_dataset import build_glove_dataset
from .bert_dataset import build_bert_dataset

from language.utils.log import get_logger


logger = get_logger()


DATASET_TYPES = {
    'word2vector': build_word2vector_dataset,
    'glove': build_glove_dataset,
    'bert': build_bert_dataset
}


def build_dataset(cfg):
    dataset_type = cfg.type
    logger.info(f'Dataset type is set to {dataset_type}.')
    if dataset_type in DATASET_TYPES:
        logger.info(f'Building dataset {cfg.name}...')
        return DATASET_TYPES[dataset_type](**cfg)
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}.')
