from ..dataset.sequence import build_sequence_dataset
from language.utils.log import get_logger


logger = get_logger()


DATASET_TYPES = {
    'sequence': build_sequence_dataset
}


def build_dataset(cfg, **kwargs):
    dataset_type = cfg.type
    logger.info(f'Dataset type is set to {dataset_type}.')
    if dataset_type in DATASET_TYPES:
        logger.info(f'Building dataset {cfg.name}...')
        return DATASET_TYPES[dataset_type](**cfg, **kwargs)
    else:
        raise ValueError(f'Invalid dataset type: {dataset_type}.')
