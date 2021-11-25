# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: inference
Description: class for driving inference.
"""

import os
import math
from multiprocessing import Lock

import numpy as np
from loguru import logger

from core.data import Tiler, TileReader
from core.utils import WSIResultRecorder, get_locations_on_heatmap, synthesize_wsi_clf_scores
from core.config.inference_base import InferenceConfig

__all__ = [
    "Inference"
]


class Inference:

    def __init__(self, configs: InferenceConfig):
        self.configs = configs
        # tiling
        self.overlap = configs.overlap
        self.gamma = configs.gamma
        # model1
        self.m1_in_size = np.array(configs.m1_input_size)
        self.m1_in_mpp = configs.m1_input_mpp
        # model2
        self.m2_in_size = np.array(configs.m2_input_size)
        self.m2_in_mpp = configs.m2_input_mpp
        # wsi clf
        self.wc_top_n = configs.wsi_clf_n
        # record
        self.output_root = configs.output_dir
        self.intermediate = False

        self.wipe_models = configs.wipe_models

    def set_wsi(self, wsi_path, group_name):
        self.wsi_path = wsi_path
        self.grp_name = group_name

    def inference(self):
        self.before_inference()
        try:
            self.do_inference()
        except Exception as e:
            logger.exception(e)
        finally:
            self.after_inference()

    def before_inference(self):
        # init Tiler and Recorder for tiling wsi and recording results
        self._lock = Lock()
        self._set_tiler()
        self._set_recorder()

    def do_inference(self):
        # model1
        self.do_model1_inference()
        self.model1_post_process()
        # model2
        self.do_model2_inference()
        self.model2_post_process()
        # wsi_clf
        self.do_wsi_clf_inference()
        self.wsi_clf_post_process()

    def after_inference(self):
        self.recorder.write_suspicious_tiles(self.tiler)
        self.recorder.write_wsi_clf_results(self.intermediate)
        if self.intermediate:
            logger.info("save intermediate results")
            self.recorder.write_intermediate()

    def do_model1_inference(self):
        model1 = self.configs.get_model1()
        # tiling setting
        self.tiler.tiling_according_to(self.m1_in_size, self.m1_in_mpp, self.overlap)
        logger.info(f"Model1 processing ...\n{self.tiler}")
        # tiler to tiles loader
        m1_dataloader = TileReader(self.tiler)
        m1_dataloader.batch_size = self.configs.m1_batch_size
        # predict
        m1_results = model1.predict_generator(m1_dataloader, verbose=1)
        # record
        self.recorder.m1_tile_bboxs = self.tiler.tiles
        self.recorder.m1_res_scores = m1_results[0].flatten()
        self.recorder.m1_res_features = m1_results[1]
        # clear session for next step
        self.wipe_models()

    def model1_post_process(self):
        logger.info("Model1 post processing ...")
        m1_scores = self.recorder.m1_res_scores
        m1_features = self.recorder.m1_res_features
        m1_tile_bboxs = self.recorder.m1_tile_bboxs
        m1_sorted_idx = np.argsort(m1_scores)[::-1]

        m2_tile_src_size = (np.array(self.m2_in_size) * self.m2_in_mpp / self.tiler.mpp).astype(int)
        # Get model1 results satisfying conditions on score threshold and numbers
        con_num_min, con_num_max = self.configs.m1_results_num_range
        con_scr_thr = self.configs.m1_score_threshold
        # initially satisfied model1 tiles and sorted idx
        nb_init_satisfied_m1_tiles = min(max(len(np.where(m1_scores > con_scr_thr)[0]), min(con_num_min, len(m1_scores))), con_num_max)
        idx_init_satisfied_m1_tiles = m1_sorted_idx[:nb_init_satisfied_m1_tiles]
        # process features
        m1_response_num = []
        m1_response_pts = []
        nb_m2_tiles = 0
        for idx in idx_init_satisfied_m1_tiles:
            m1_feature = m1_features[idx][..., 1]  # first dimension on axis -1 is the positive indicator
            locations = np.stack(get_locations_on_heatmap(m1_feature.copy(), self.m1_in_size))
            locations = (
                locations * self.m1_in_mpp / self.tiler.mpp +
                m1_tile_bboxs[idx][:2] -
                (np.array(self.m2_in_size) * self.m2_in_mpp / self.tiler.mpp) / 2
            ).astype(int)  # global top-left of m2 input image
            # restrict number of model2 tiles to m1_results_num_range
            nb_m2_tiles += len(locations)
            if nb_m2_tiles > con_num_max:
                break
            m1_response_num.append(len(locations))
            m1_response_pts.append(locations)
        # final satisfied model1 tiles
        nb_satisfied_m1_tiles = len(m1_response_num)
        idx_satisfied_m1_tiles = idx_init_satisfied_m1_tiles[:nb_satisfied_m1_tiles]
        logger.info(f"satisfied model1 tiles: {nb_satisfied_m1_tiles}")
        # get model2 bbox and read level
        m2_in_tl_pts = np.concatenate(m1_response_pts, axis=0)
        m2_tile_bboxs = np.concatenate((m2_in_tl_pts, np.ones_like(m2_in_tl_pts) * m2_tile_src_size), axis=-1)
        logger.info(f"response points: {len(m2_in_tl_pts)}")
        # record
        self.recorder.m1_response_num = m1_response_num
        self.recorder.m1_response_idx = idx_satisfied_m1_tiles

        self.recorder.m2_tile_bboxs = m2_tile_bboxs

    def do_model2_inference(self):
        model2 = self.configs.get_model2()
        # tiling setting
        read_level = max(0, int(math.log(self.m2_in_mpp/self.tiler.mpp, self.tiler.level_ratio)))
        self.tiler.tiling_set_to(self.recorder.m2_tile_bboxs, self.m2_in_size, self.m2_in_mpp, read_level)
        logger.info(f"Model2 processing ...\n{self.tiler}")
        # tiler to tiles loader
        m2_dataloader = TileReader(self.tiler)
        m2_dataloader.batch_size = self.configs.m2_batch_size
        # predict
        m2_results = model2.predict_generator(m2_dataloader, verbose=1)
        # record
        self.recorder.m2_res_scores = m2_results[0].flatten()
        self.recorder.m2_res_features = m2_results[1]
        # clear session for next step
        self.wipe_models()

    def model2_post_process(self):
        logger.info("Model2 post processing ...")
        m2_scores = self.recorder.m2_res_scores
        m2_sorted_idx = np.argsort(m2_scores)[::-1]

        self.recorder.most_suspicious_idx = m2_sorted_idx[:self.configs.most_suspicious_n]
        self.recorder.wc_top_idx = m2_sorted_idx[:max(self.configs.wsi_clf_n)]

    def do_wsi_clf_inference(self):
        m2_features = self.recorder.m2_res_features
        wc_top_idx = self.recorder.wc_top_idx
        wc_scores = {}
        logger.info(f"WSI classification processing ...")
        for top_n in self.wc_top_n:
            wc_scores[top_n] = []
            for id in range(2):
                wsi_clf = self.configs.get_wsi_clf(top_n, id)
                score = wsi_clf.predict(np.stack([m2_features[wc_top_idx[:top_n]]])).flatten()[0]
                wc_scores[top_n].append(float(score))
                # clear session for next step
                self.wipe_models()
        self.recorder.wc_scores = wc_scores

    def wsi_clf_post_process(self):
        wsi_score = synthesize_wsi_clf_scores(self.recorder.wc_scores)
        self.recorder.wsi_score = float(wsi_score)
        logger.info(f"WSI classification: {wsi_score}")

    def _set_tiler(self):
        self.tiler = Tiler(self.wsi_path, self.gamma)
        self.tiler.set_multiprocess_lock(self._lock)

    def _set_recorder(self):
        self.results_root = os.path.join(self.output_root, self.grp_name, os.path.basename(self.wsi_path).split('.')[0])
        self.recorder = WSIResultRecorder(self.results_root)

