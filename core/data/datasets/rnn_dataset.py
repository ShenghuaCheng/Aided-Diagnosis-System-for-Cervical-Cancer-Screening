# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: rnn_dataset
Description: dataset for rnn related tasks.
"""

import random

import numpy as np
import pandas as pd
from loguru import logger
from xlrd import XLRDError

from core.utils.dataset_tools import process_df_config, _get_img_ls

__all__ = [
    "RnnDataset"
]


class RnnDataset:
    def __init__(
            self,
            config_file,
    ):
        self.config_file = config_file
        # parse
        self._parse_config()

    def _parse_config(self):
        self.df_config = {"train": None, "test": None, "val": None}
        for k in self.df_config:
            try:
                self.df_config[k] = pd.read_excel(self.config_file, sheetname=k)
            except XLRDError as e:
                logger.warning(f"No {k} set.")
                logger.warning(e)
                continue
            self.df_config[k] = process_df_config(self.df_config[k])

    def _get_wsi_pools(self):
        self.wsi_pools = {}
        # get wsi pool
        for k, df_cfg in self.df_config.items():
            if df_cfg is None:
                self.wsi_pools[k] = None
                continue
            wsi_pools = []
            for r_id, row in df_cfg.iterrows():
                index_files = row["index_files"]
                wsi_pool = _get_img_ls(index_files)
                wsi_pools.append(wsi_pool)
            self.wsi_pools[k] = wsi_pools

    def get_set(self, name="train"):
        """ Get set
            return config list, image pool, mask dict
        """
        if not hasattr(self, 'wsi_pools'):
            self._get_wsi_pools()

        return self.df_config[name].to_dict('list'), self.wsi_pools[name]

    def sampling(self, name="train"):
        """ Return sampled idx pools
        """
        config, wsi_pools = self.get_set(name)
        sampled_idx_pools = []
        for i in range(len(wsi_pools)):
            nb_smp = int(config["subset_sample"][i])
            pl_smp = wsi_pools[i]
            # sampling
            sample_idx = random.sample(list(range(len(pl_smp))), nb_smp)
            sampled_idx_pools.append(sorted(sample_idx))
        return sampled_idx_pools

    def write_config(self, path):
        writer = pd.ExcelWriter(path)
        for k, v in self.df_config.items():
            if v is None:
                continue
            v.to_excel(writer, sheet_name=k, index=False)
        writer.close()
        logger.info(f"write new config of dataset to {path}")

    def __repr__(self):
        repr_str = ""
        print(repr_str)


