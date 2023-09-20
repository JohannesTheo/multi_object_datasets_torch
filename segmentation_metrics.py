import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


def flatten_and_one_hot(true_groups):
    r"""Reshapes a batch of masks as returned from the dataset/loader to the format requested by the ari method.

    Input shape:  [batch_size, max_num_entities, channels, height, width]
    Output shape: [batch_size, (channels*height*width), max_num_entities] or in ari terms:
                  [batch_size, n_points, n_true_groups]
    """
    batch_size, max_num_entities, channels, height, width = true_groups.shape
    desired_shape = [batch_size, channels * height * width, max_num_entities]

    true_groups_oh = torch.permute(true_groups, [0, 2, 3, 4, 1])
    true_groups_oh = torch.reshape(true_groups_oh, desired_shape)

    return true_groups_oh


def random_predictions_like(true_groups, encoding: str):
    r"""This method takes a batch of masks from the dataset/loader and returns a matching batch of
      random predictions in the format requested by 'encoding'. Can be either *one-hot* for the ari
      method or *categorical* for the sc scores.

    Input shape:              [batch_size, max_num_entities, channels, height, width]
    Output shape onehot:      [batch_size, n_points, n_true_groups]
    Output shape categorical: [batch_size, 1, height, width]
    """
    assert encoding in ['oh', 'onehot', 'one-hot', 'cat', 'categorical'], f"Unsupported encoding: {encoding}. " \
                                                                          f"Must be one of ['oh','onehot','one-hot'] " \
                                                                          f"or ['cat','categorical']."

    if encoding in ['oh', 'onehot', 'one-hot']:
        batch_size, max_num_entities, channels, height, width = true_groups.shape
        desired_shape = [batch_size, height * width, max_num_entities]

        random_prediction = torch.randint(low=0, high=max_num_entities, size=desired_shape[:-1])
        random_prediction = F.one_hot(random_prediction, max_num_entities)

        return random_prediction
    elif encoding in ['cat', 'categorical']:
        return torch.argmax(torch.rand(true_groups.shape), dim=1)


# tf-to-torch port of:
# https://github.com/deepmind/multi_object_datasets/blob/9c670cd630940b9f8f5b0e9728472201a50a3370/segmentation_metrics.py#L20
# We added the ignore_background argument to be consistent with the other metrics methods
def adjusted_rand_index(true_mask, pred_mask, ignore_background=False, name='ari_score'):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.

      This implementation ignores points with no cluster label in `true_mask` (i.e.
      those points for which `true_mask` is a zero vector). In the context of
      segmentation, that means this function can ignore points in an image
      corresponding to the background (i.e. not to an object).

      Args:
        true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
          The true cluster assignment encoded as one-hot.
        pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
          The predicted cluster assignment encoded as categorical probabilities.
          This function works on the argmax over axis 2.
        name: str. Name of this operation (defaults to "ari_score"). # unused as of now in torch version

      Returns:
        ARI scores as a torch.float32 `Tensor` of shape [batch_size].

      Raises:
        ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
          We've chosen not to handle the special cases that can occur when you have
          one cluster per datapoint (which would be unusual).

      References:
        Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
          https://link.springer.com/article/10.1007/BF01908075
        Wikipedia
          https://en.wikipedia.org/wiki/Rand_index
        Scikit Learn
          http://scikit-learn.org/stable/modules/generated/\
          sklearn.metrics.adjusted_rand_score.html
    """
    if ignore_background:
        true_mask = true_mask[...,1:]

    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
      # This rules out the n_true_groups == n_pred_groups == n_points
      # corner case, and also n_true_groups == n_pred_groups == 0, since
      # that would imply n_points == 0 too.
      # The sklearn implementation has a corner-case branch which does
      # handle this. We chose not to support these cases to avoid counting
      # distinct clusters just to check if we have one cluster per datapoint.
      raise ValueError(
          "adjusted_rand_index requires n_groups < n_points. We don't handle "
          "the special cases that can occur when you have one cluster "
          "per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    # We convert true and predicted clusters to one-hot ('oh') representations.
    true_mask_oh = true_mask.to(dtype=torch.float32)  # already one-hot
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(dtype=torch.float32)

    n_points = torch.sum(true_mask_oh, dim=[1, 2])

    nij = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(nij, dim=1)
    b = torch.sum(nij, dim=2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points*(n_points-1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    # The case where n_true_groups == n_pred_groups == 1 needs to be
    # special-cased (to return 1) as the above formula gives a divide-by-zero.
    # This might not work when true_mask has values that do not sum to one:
    both_single_cluster = torch.logical_and(
      _all_equal(true_group_ids), _all_equal(pred_group_ids)
    )
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


# tf-to-torch port of:
# https://github.com/deepmind/multi_object_datasets/blob/9c670cd630940b9f8f5b0e9728472201a50a3370/segmentation_metrics.py#L95
def _all_equal(values):
    """Whether values are all equal along the final axis."""
    return torch.all(torch.eq(values, values[..., :1]), dim=-1)


# source: https://github.com/applied-ai-lab/genesis/blob/9abf202bbad6fa4a675117fdea0be163e4f16695/utils/misc.py#L162
def iou_binary(mask_A, mask_B):
    assert mask_A.shape == mask_B.shape
    assert mask_A.dtype == torch.bool
    assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())


# source: https://github.com/applied-ai-lab/genesis/blob/9abf202bbad6fa4a675117fdea0be163e4f16695/utils/misc.py#L173
def average_segcover(segA, segB, ignore_background=False):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]

    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.

    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    nonignore = (segA >= 0)

    mean_scores = torch.tensor(bsz*[0.0])
    N = torch.tensor(bsz*[0])
    scaled_scores = torch.tensor(bsz*[0.0])
    scaling_sum = torch.tensor(bsz*[0])

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA > 0]).tolist()
    else:
        iter_segA = torch.unique(segA[segA >= 0]).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0])
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) * nonignore
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension
    # mean_sc is mSC, scaled_sc is SC (scaled by mask size)
    #return mean_sc.mean(0), scaled_sc.mean(0)
    return mean_sc, scaled_sc


# source: https://github.com/pairlab/SlotFormer/blob/5d97e8779aa98ffdfb3d5506accb6bf110b5cac4/slotformer/video_prediction/vp_utils.py#LL225C1-L243C50
def hungarian_miou(gt_mask, pred_mask, ignore_background=False):
    """both mask: [H*W] after argmax, 0 is gt background index."""

    if ignore_background:
        true_oh = F.one_hot(gt_mask).float()[..., 1:]  # only foreground, [HW, N]
    else:
        true_oh = F.one_hot(gt_mask).float() # keep background mask, [HW, N]
    pred_oh = F.one_hot(pred_mask).float()  # [HW, M]
    N, M = true_oh.shape[-1], pred_oh.shape[-1]
    # compute all pairwise IoU
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)  # [N, M]
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None, :]  # [N, M]
    iou = intersect / (union + 1e-8)  # [N, M]
    iou = iou.detach().cpu().numpy()
    # find the best match for each gt
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    # there are two possibilities here
    #   1. M >= N, just take the best match mean
    #   2. M < N, some objects are not detected, their iou is 0
    if M >= N:
        assert (row_ind == np.arange(N)).all()
        return iou[row_ind, col_ind].mean()
    return iou[row_ind, col_ind].sum() / float(N)
