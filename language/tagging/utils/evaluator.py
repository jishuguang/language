from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

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
        self._setup_data(val_dataset)
        self._tag_vocab = val_dataset.get_tag_vocab()

    def _setup_data(self, val_dataset):
        batch_size = self._cfg.device.batch_size
        num_worker = self._cfg.device.num_worker
        logger.info(f'batch_size: {batch_size}')
        logger.info(f'num_workers: {num_worker}')
        self._total_steps = len(val_dataset) // batch_size
        self._train_iter = DataLoader(val_dataset, shuffle=False,
                                      batch_size=batch_size, num_workers=num_worker,
                                      drop_last=False)

    def evaluate(self, model):
        model.eval()
        logger.info(f'Evaluating...')
        ground_truth = list()
        prediction = list()
        for data in tqdm(self._train_iter, total=self._total_steps):
            self._to_device(data)
            with torch.no_grad():
                predict_tag_index = model.forward_infer(data).to(torch.long).tolist()
            tag_index = data['tags'].to(torch.long).tolist()
            valid_len = data['valid_len'].to(torch.long).tolist()
            length = len(valid_len)
            for i in range(length):
                # ignore the first tag, which corresponds with '<cls>'
                prediction.append(self._tag_vocab.lookup_tokens(predict_tag_index[i][1:1 + valid_len[i]]))
                ground_truth.append(self._tag_vocab.lookup_tokens(tag_index[i][1:1 + valid_len[i]]))
        logger.info(f'\n{classification_report(ground_truth, prediction)}')
        return {
            'precision': precision_score(ground_truth, prediction),
            'recall': recall_score(ground_truth, prediction),
            'f1score': f1_score(ground_truth, prediction)
        }

    def _to_device(self, data):
        """Move a batch to specified device."""
        device = self._cfg.device.name
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
