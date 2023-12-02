"""
A two-view sparse feature matching pipeline on triplets.

If a triplet is found, runs the extractor on three images and
then runs matcher/filter/solver for all three pairs.

Losses and metrics get accumulated accordingly.

If no triplet is found, this falls back to two_view_pipeline.py
"""

import torch

from .matchers.nearest_neighbor_matcher import NearestNeighborMatcher
from ..utils.misc import get_twoview, stack_twoviews, unstack_twoviews
from .two_view_pipeline import TwoViewPipeline

import torch.nn.functional as F
def has_triplet(data):
    # we already check for image0 and image1 in required_keys
    return "view2" in data.keys()


class FriendsTripletPipeline(TwoViewPipeline):
    default_conf = {"help_view": 1, **TwoViewPipeline.default_conf}

    def create_help_descritors(self, pred, gt_views_index):
        """处理函数，生成视图对齐的辅助描述符"""
        help_descriptors = pred["descriptors0"].clone()
        index = pred[f"gt_{gt_views_index[0]}_{gt_views_index[1]}_matches0"]
        B_descriptors = pred[f"descriptors{gt_views_index[1]}"]

        for b in range(0, index.shape[0]):
            for i in range(0, index.shape[1]):
                if index[b][i] != -1:
                    help_descriptors[b][i] = B_descriptors[b][index[b][i]]

        pred.update({"help_descriptors": help_descriptors})
        return pred

    def _forward(self, data):
        assert has_triplet(data)

        pred2 = self.extract_view(data, "2")
        key_list = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1",
                    "descriptors0", "descriptors1"]

        pred = {
            **dict((key, data[key]) for key in key_list),
            **{k + "2": v for k, v in pred2.items()}
        }


        if self.conf.help_view == 1:
            gt_views_idx = [0, 1]  # train match 0-2
            match_idx = [0, 2]
        else:
            gt_views_idx = [0, 2]
            match_idx = [0, 1]

        pred.update({"view_idx": match_idx})

        pred.update({f"gt_{gt_views_idx[0]}_{gt_views_idx[1]}_matches0": data["matches0"]})
        pred = self.create_help_descritors(pred, gt_views_idx)


        # # assert not self.conf.run_gt_in_forward
        # pred0 = self.extract_view(data, "0")
        # pred1 = self.extract_view(data, "1")
        # pred2 = self.extract_view(data, "2")
        #
        # pred = {}
        # pred = {
        #     **{k + "0": v for k, v in pred0.items()},
        #     **{k + "1": v for k, v in pred1.items()},
        #     **{k + "2": v for k, v in pred2.items()},
        # }

        # 获取两视图的匹配真值

        # if self.conf.help_view == 1:
        #     gt_views_idx = [0, 1]  # train match 0-2
        #     match_idx = [0, 2]
        # else:
        #     gt_views_idx = [0, 2]
        #     match_idx = [0, 1]
        #
        # pred.update({"view_idx": match_idx})

        # 获取0-2匹配
        # nn_result = nn(pred)

        # pred.update({f"gt_{gt_views_idx[0]}_{gt_views_idx[1]}_{k}": v for k, v in nn_result.items()})
        # pred = self.create_help_descritors(pred, gt_views_idx)

        # if self.conf.ground_truth.name:  # train
        #     gt_pred = self.ground_truth({**data, **pred})
        #     pred.update({f"gt_{gt_views_idx[0]}_{gt_views_idx[1]}_{k}": v for k, v in gt_pred.items()})
        #     pred = self.create_help_descritors(pred, gt_views_idx)
        # else:  # eval

            # # 默认本身真值
        # pred.update({"help_descriptors": pred["descriptors0"].clone()})
        # pred = {**pred, **self.matcher({**data, **pred, "help_view": 1})}   # 执行0-2匹配






                # 执行一次匹配

                # 获取匹配结果作为假真值

                # 在预测中删除第一次匹配的结果

                # 获取help

                # 执行新的匹配


        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred, "help_view": self.conf.help_view})}
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
        if self.conf.help_view == 1:
            gt_views_idx = [0, 2]  # train match 0-2
        else:
            gt_views_idx = [0, 1]
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred, "view_idx": gt_views_idx})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

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
