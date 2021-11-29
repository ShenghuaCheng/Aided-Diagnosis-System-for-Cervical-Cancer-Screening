# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: inference_tools
Description: useful tools for inference
"""

import numpy as np
from skimage import measure

__all__ = [
    "get_locations_on_heatmap",
    "synthesize_wsi_clf_scores",
    "remove_overlap_results"
]


def get_locations_on_heatmap(heatmap, output_shape, threshold=0.7):
    max_prob = np.max(heatmap)
    heatmap[heatmap < max_prob * threshold] = 0  # mask out area lower than related threshold
    mask = (heatmap != 0).astype(np.uint8)
    mask_label = measure.label(mask.copy())
    locations = []
    for c in range(mask_label.max()):
        # process per connected domains
        heatmap_temp = heatmap * (mask_label == (c+1))
        a = np.where(heatmap_temp == heatmap_temp.max())  # 0 for row_id(y), 1 for col_id(x)
        # get center point of roi
        c_y = np.around((a[0][0] + 0.5) * output_shape[1] / heatmap.shape[0]).astype(int)
        c_x = np.around((a[1][0] + 0.5) * output_shape[0] / heatmap.shape[1]).astype(int)
        locations.append((c_x, c_y))
    return locations


def synthesize_wsi_clf_scores(score_dict):
    """ final score of wsi clf
    """
    avgs = [np.mean(v) for _, v in score_dict.items()]
    avg_all = np.mean(avgs)
    avg_max = np.max(avgs)
    avg_min = np.min(avgs)
    avg_std = np.min(avgs)
    final_score = avg_all if avg_std > 0.15 else avg_min if avg_all < 0.15 else avg_max
    return final_score


def remove_overlap_results(scores, points, mpp, distance_threshold=200):
    """ Remove overlap results, default distance is 200um
        input requiring ndarray
        return a flags of reserving
    """
    dist_pix_threshold = distance_threshold / mpp
    sorted_idx = np.argsort(scores)[::-1]
    # remove overlap
    reserving_flg = [True] * len(scores)  # mark removed idx as False
    for cur_i, cur_idx in enumerate(sorted_idx):
        cur_point = points[cur_idx]
        if not reserving_flg[cur_idx]:
            # cur idx has been marked as removed
            continue
        # compare with points score less than it
        for idx in sorted_idx[cur_i:]:
            if not reserving_flg[idx] or idx == cur_idx:
                # idx is itself or
                # idx has been marked as removed
                continue
            tmp_point = points[idx]
            dist_pix = np.sqrt(np.sum(np.power(cur_point-tmp_point, 2)))
            if dist_pix < dist_pix_threshold:
                reserving_flg[idx] = False
    return reserving_flg
