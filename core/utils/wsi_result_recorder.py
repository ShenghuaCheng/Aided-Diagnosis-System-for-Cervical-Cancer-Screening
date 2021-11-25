# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: wsi_result_recorder
Description: recorder for recording wsi inference results
"""

import os
import json
import cv2
from ..common import check_dir

__all__ = [
    "WSIResultRecorder"
]


class WSIResultRecorder:

    def __init__(self, save_root):
        self.save_root = save_root
        check_dir(save_root, True)
        # Model1 related
        self.m1_tile_bboxs = None
        self.m1_res_scores = None
        self.m1_res_features = None
        # results
        self.m1_response_idx = None
        self.m1_response_num = None

        # Model2 related
        self.m2_tile_bboxs = None
        self.m2_res_scores = None
        self.m2_res_features = None
        # results
        self.most_suspicious_idx = None

        # WSI clf related
        self.wc_top_idx = None
        self.wc_scores = None
        # results
        self.wsi_score = None

    def write_suspicious_tiles(self, tiler):
        save_dir = os.path.join(self.save_root, "suspicious_areas")
        check_dir(save_dir, True)
        for i, idx in enumerate(self.most_suspicious_idx):
            image = tiler[idx]
            score = self.m2_res_scores[idx]
            save_path = os.path.join(save_dir, "{:0>2d}_{:.6f}.jpg".format(i, score))
            cv2.imwrite(save_path, image[..., ::-1])

    def write_wsi_clf_results(self, intermediate=False):
        wsi_clf_res = {"final": self.wsi_score}
        if intermediate:
            wsi_clf_res.update(self.wc_scores)
        f = open(os.path.join(self.save_root, "wsi_clf.json"), "w+")
        json.dump(wsi_clf_res, f, indent=2)
        f.close()

    def write_intermediate(self):
        # you can save any intermediate results you want
        pass