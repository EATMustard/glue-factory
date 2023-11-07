import torch
import torch.nn as nn
from omegaconf import OmegaConf


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights  # 16，513，513

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)    # 16,计算16个数据中每个的没有匹配的数量
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)   # 16,计算16个数据中每个有匹配的数量

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2)) # 求所有匹配的loss
    nll_pos /= num_pos.clamp(min=1.0)   # 平均

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)  # 两个负Loss
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1) # 平均

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"] # 16,513,513
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg     # 0.5*pos+0.5*neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,     # 其实是取的两个的均值
            },
        )

    def nll_loss(self, log_assignment, data):
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)   # 512,512
        positive = data["gt_assignment"].float()    # 16,512,512
        neg0 = (data["gt_matches0"] == -1).float()  # 16,512    1的是没有匹配的特征点 0 是有匹配的
        neg1 = (data["gt_matches1"] == -1).float()  # 16,512

        weights = torch.zeros_like(log_assignment)  # 16,513，513
        weights[:, :m, :n] = positive   # 前512行512列

        weights[:, :m, -1] = neg0   # 前m行的最后一列
        weights[:, -1, :m] = neg1
        return weights  # 一个完整的分配矩阵
