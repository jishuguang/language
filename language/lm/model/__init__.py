from .word2vector import SkipGram
from .glove import Glove
from .bert import Bert
from language.utils.log import get_logger


logger = get_logger()


MODELS = {
    'skip_gram': SkipGram,
    'glove': Glove,
    'bert': Bert
}


def build_model(cfg, **kwargs):
    name = cfg.name
    if name in MODELS:
        logger.info(f'Build embedding model {name}...')
        kwargs.update(cfg)
        return MODELS[name](**kwargs)
    else:
        raise ValueError(f'Invalid embedding model name: {name}.')
