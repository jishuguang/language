import os

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data

from language.utils.log import get_logger
from language.utils.serialization import save_model

logger = get_logger()


class Trainer:

    def __init__(self, train_cfg, evaluator, model, train_data, save_dir):
        self._cfg = train_cfg
        self._evaluator = evaluator
        self._model = model.to(train_cfg.device.name)
        self._setup_data(train_data)
        self._setup_scheduler()
        self._save_dir = save_dir
        self._key_metric = train_cfg.key_metric

    def _setup_data(self, train_data):
        batch_size = self._cfg.device.batch_size
        num_worker = self._cfg.device.num_worker
        logger.info(f'batch_size: {batch_size}')
        logger.info(f'num_workers: {num_worker}')
        self._total_steps = len(train_data) // batch_size
        self._train_iter = data.DataLoader(train_data, shuffle=True,
                                           batch_size=batch_size, num_workers=num_worker,
                                           drop_last=True)

    def _setup_scheduler(self):
        learn_paras = {
            'lr': self._cfg.learn.lr,
        }
        self._optimizer = getattr(optim, self._cfg.learn.method)(self._model.parameters(), **learn_paras)
        self._scheduler = MultiStepLR(self._optimizer, self._cfg.learn.milestones)

    def train(self):
        best_key_metric = 0
        for epoch in range(self._cfg.learn.epochs):
            self._train_one_epoch(epoch)
            metric_dict = self._evaluator.evaluate(self._model)
            msg = f'Epoch {epoch}: {" | ".join([f"{key}: {value:.4f}" for key, value in metric_dict.items()])}.'
            key_metric = metric_dict[self._key_metric]
            logger.info(msg)
            if key_metric >= best_key_metric:
                best_key_metric = key_metric
                # save model
                save_model(self._model, os.path.join(self._save_dir, 'model', 'model_best.pth'))
                # save evaluation result
                with open(os.path.join(self._save_dir, 'model', 'evaluation.txt'), 'a') as f:
                    f.write(msg + os.linesep)
            save_model(self._model, os.path.join(self._save_dir, 'model', 'model_last.pth'))

            self._scheduler.step()

    def _train_one_epoch(self, epoch):
        step = 0
        self._model.train()
        for train_batch in self._train_iter:
            self._optimizer.zero_grad()
            self._to_device(train_batch)
            loss = self._model.forward_train(train_batch)
            logger.info(f'[Epoch {epoch}][Step {step}/{self._total_steps}] '
                        f'lr: {self._scheduler.get_last_lr()[0]:.5f}'
                        f' | loss {loss:.4f}')
            loss.backward()
            self._optimizer.step()
            step += 1

    def _to_device(self, train_batch):
        """Move a batch to specified device."""
        device = self._cfg.device.name
        for key, value in train_batch.items():
            if isinstance(value, torch.Tensor):
                train_batch[key] = value.to(device)
