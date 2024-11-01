import torch
from torch import nn


class ZilnLoss:
    def __init__(self, label_clip_val=0., class_threshold=0.):
        self.label_clip_val = label_clip_val
        self.class_threshold = class_threshold

    def _ziln_loss(self, logits, label):
        if self.label_clip_val > 0:
            label = torch.clamp(label, min=0, max=self.label_clip_val)

        p_pred = logits[:, 0]
        p_ground = (label > self.class_threshold).to(torch.float32)

        cls_loss = nn.BCEWithLogitsLoss(reduction='none')(p_pred, p_ground)

        mu = logits[:, 1]
        sigma = torch.maximum(nn.Softplus()(logits[:, 2]), torch.sqrt(torch.tensor(torch.finfo(torch.float32).eps)))

        safe_labels = p_ground * label + (1 - p_ground) * torch.ones_like(label)
        reg_loss = 0. - p_ground * torch.distributions.LogNormal(mu, sigma).log_prob(safe_labels)

        return torch.mean(cls_loss + reg_loss, dim=0)

    def _ziln_predict(self, logits, pred_clip_val=0):
        p, mu, sigma = logits[:, 0], logits[:, 1], logits[:, 2]
        p = torch.nn.Sigmoid()(p)
        sigma = torch.nn.Softplus()(sigma)

        preds = (p * torch.exp(mu + 0.5 * torch.square(sigma)))

        # # 空值补0
        # is_nan = torch.isnan(preds)
        # padding_preds = torch.zeros_like(preds)
        # preds = torch.where(is_nan, padding_preds, preds)

        if pred_clip_val > 0:
            preds = torch.clamp(preds, min=0, max=pred_clip_val)

        return p, preds


class MSELoss:
    def __init__(self):
        pass