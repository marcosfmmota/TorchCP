import torch

from torchcp.classification.predictors import SplitPredictor
from torchcp.utils.common import calculate_conformal_value


class NoiseRobustPredictor(SplitPredictor):
    """
    Noise Robust adaptation of base SplitPredictor (Penso and Goldberger, 2024)
    Paper: http://arxiv.org/abs/2405.02648

    :param score_function: non-conformity score function.
    :param noise_level: estimate of dataset label noise level
    :param model: a pytorch model
    :param temperature: the temperature of Temperature Scaling
    """

    def __init__(self, score_function, noise_level, model=None, temperature=1):
        super().__init__(score_function, model, temperature)
        self.noise_level = noise_level

    def calibrate(self, cal_dataloader, alpha):
        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._logits_transformation(self._model(tmp_x)).detach()
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        scores = self._robust_score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _robust_score_function(self, logits, labels):
        scores_all = self.score_function(logits)
        data_idxs = torch.arange(scores_all.shape[0])
        label_idxs = labels.squeeze().int()
        scores_noisy = scores_all[data_idxs, label_idxs]
        weigthed_scores = (1 / scores_all.shape[-1]) * torch.sum(scores_all, dim=1)
        robust_scores = (1 - self.noise_level) * scores_noisy + self.noise_level * weigthed_scores
        return robust_scores
