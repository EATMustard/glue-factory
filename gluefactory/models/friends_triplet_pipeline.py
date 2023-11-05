"""
A two-view sparse feature matching pipeline on triplets.

If a triplet is found, runs the extractor on three images and
then runs matcher/filter/solver for all three pairs.

Losses and metrics get accumulated accordingly.

If no triplet is found, this falls back to two_view_pipeline.py
"""

import torch

from ..utils.misc import get_twoview, stack_twoviews, unstack_twoviews
from .two_view_pipeline import TwoViewPipeline


def has_triplet(data):
    # we already check for image0 and image1 in required_keys
    return "view2" in data.keys()


class FriendsTripletPipeline(TwoViewPipeline):
    default_conf = {**TwoViewPipeline.default_conf}

    def create_help_descritors(self, pred):
        """处理函数，生成视图对齐的辅助描述符"""
        help_descriptors = pred["descriptors0"].clone()
        index = pred["gt_matches0_0_1"]
        B_descriptors = pred["descriptors1"]

        for b in range(0, index.shape[0]):
            for i in range(0, index.shape[1]):
                if index[b][i] != -1:
                    help_descriptors[b][i] = B_descriptors[b][index[b][i]]

        pred.update({"help_descriptors": help_descriptors})
        return pred

    def _forward(self, data):
        assert has_triplet(data)

        # assert not self.conf.run_gt_in_forward
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred2 = self.extract_view(data, "2")

        pred = {}
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
            **{k + "2": v for k, v in pred2.items()},
        }

        # 获取两视图的匹配真值
        gt_views_idx = [0, 1]
        if self.conf.ground_truth.name:
            gt_pred = self.ground_truth({**data, **pred, "view_idx": gt_views_idx})
            pred.update({f"gt_{k}_{gt_views_idx[0]}_{gt_views_idx[1]}": v for k, v in gt_pred.items()})

        pred = self.create_help_descritors(pred)

        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        return pred

    def loss(self, pred, data):
        assert has_triplet(data)
        losses = {}
        metrics = {}
        total = 0

        # get labels
        gt_views_idx = [0, 2]
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred, "view_idx": gt_views_idx})
            pred.update({f"gt_{k}_{gt_views_idx[0]}_{gt_views_idx[1]}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
