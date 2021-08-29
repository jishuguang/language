from .bert import Bert
from language.utils.log import get_logger


logger = get_logger()


MODELS = {
    'bert': Bert
}


def build_model(cfg, **kwargs):
    name = cfg.name
    if name in MODELS:
        logger.info(f'Build tagging model {name}...')
        kwargs.update(cfg)
        return MODELS[name](**kwargs)
    else:
        raise ValueError(f'Invalid tagging model name: {name}.')
