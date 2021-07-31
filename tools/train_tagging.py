import argparse
import os
import time
from logging import FileHandler

from language.tagging.dataset import build_dataset
from language.tagging.model import build_model
from language.tagging.utils.trainer import Trainer
from language.tagging.utils.evaluator import Evaluator
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
    elif 'bert' in pretrain_cfg:
        model.load_bert_model(pretrain_cfg.bert)

    return model


def train(cfg):
    train_dataset = build_dataset(cfg.data, purpose='train')
    val_dataset = build_dataset(cfg.data, purpose='val')
    model = build_model(cfg.model, vocab=train_dataset.get_vocab(), tag_vocab=train_dataset.get_tag_vocab())
    load_pretrain_model(model, cfg.pretrain)
    evaluator = Evaluator(cfg.evaluator, val_dataset)
    Trainer(cfg.trainer, evaluator, model, train_dataset, cfg.save.dir).train()


def main():
    parser = argparse.ArgumentParser(description='Train sequence tagging model.')
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
