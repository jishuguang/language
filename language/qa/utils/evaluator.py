from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from language.utils.log import get_logger


logger = get_logger()


class Evaluator:

    def __init__(self, evaluator_cfg, val_dataset):
        """
        :param evaluator_cfg: dict, evaluator config.
        :param val_dataset: torch dataset.
        """
        logger.info(f'Setup evaluator...')
        self._cfg = evaluator_cfg
        self._impossible = evaluator_cfg.impossible
        self._setup_data(val_dataset)

    def _setup_data(self, val_dataset):
        batch_size = self._cfg.device.batch_size
        num_worker = self._cfg.device.num_worker
        logger.info(f'batch_size: {batch_size}')
        logger.info(f'num_workers: {num_worker}')
        self._total_steps = len(val_dataset) // batch_size
        self._val_iter = DataLoader(val_dataset, shuffle=False,
                                    batch_size=batch_size, num_workers=num_worker,
                                    drop_last=False)

    def evaluate(self, model):
        model.eval()
        logger.info(f'Evaluating...')
        total_amount = 0
        correct = 0
        for data in tqdm(self._val_iter, total=self._total_steps):
            self._to_device(data)
            with torch.no_grad():
                no_answer, start_pred, end_pred = model.forward_infer(data)
            start = data['start']
            end = data['end']
            for i in range(start.shape[0]):
                total_amount += 1
                if self._impossible and no_answer[i]:
                    if start[i, 0] == 0:
                        correct += 1
                else:
                    if any([s == start_pred[i] and e == end_pred[i] for s, e in zip(start[i], end[i])]):
                        correct += 1
        em = correct / total_amount
        return {'EM': em}

    def _to_device(self, data):
        """Move a batch to specified device."""
        device = self._cfg.device.name
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
