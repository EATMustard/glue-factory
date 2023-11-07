import torch

from ..utils.tensor import batch_to_device
from .viz2d import cm_RdGn, plot_heatmaps, plot_image_grid, plot_keypoints, plot_matches


def make_match_figures(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []

    if "view_idx" in pred_:
        view_index = pred_["view_idx"]
        pred_.pop("view_idx")

    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    if "view2" in data and view_index == [0, 2]:
        view0, view1 = data["view0"], data["view2"]
        kp0, kp1 = pred["keypoints0"], pred["keypoints2"]
    else:
        view0, view1 = data["view0"], data["view1"]
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}
