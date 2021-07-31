import argparse
import os
import time
from logging import FileHandler

from language.lm.dataset import build_dataset
from language.lm.model import build_model
from language.lm.utils.trainer import Trainer
from language.utils.log import get_logger
from language.utils.config import get_cfg_defaults
from language.utils.serialization import load_model


logger = get_logger()


def load_pretrain_model(model, pretrain_cfg):
    if pretrain_cfg is None:
        return model

    if 'resume' in pretrain_cfg:
        load_model(model, pretrain_cfg.resume, resume=True)
    elif 'load' in pretrain_cfg:
        load_model(model, pretrain_cfg.load)

    return model


def train(cfg):
    dataset = build_dataset(cfg.data)
    model = build_model(cfg.model, vocab=dataset.get_vocab())
    load_pretrain_model(model, cfg.pretrain)
    Trainer(cfg.trainer, model, dataset, cfg.save.dir).train()


def main():
    parser = argparse.ArgumentParser(description='Train language model.')
    parser.add_argument('--config', required=True, type=str, help='Path to config.')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.save.dir = os.path.join(cfg.save.dir, time.strftime("%Y%m%d%H%M%S"))
    os.makedirs(cfg.save.dir, exist_ok=True)
    cfg.freeze()

    logger.addHandler(FileHandler(os.path.join(cfg.save.dir, f'train.log')))
    logger.info(f'Loading config {args.config}.')
    logger.info(f'Config:\n {cfg}')
    train(cfg)


if __name__ == '__main__':
    main()
