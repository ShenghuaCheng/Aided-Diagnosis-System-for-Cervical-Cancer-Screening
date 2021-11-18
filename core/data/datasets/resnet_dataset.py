# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: resnet_dataset
Description: dataset for resnet related tasks.
"""

import os
import random

import numpy as np
import pandas as pd
from loguru import logger
from xlrd import XLRDError

__all__ = [
    "ResnetDataset"
]


class ResnetDataset:
    def __init__(
            self,
            config_file,
            with_mask=False,
    ):
        self.config_file = config_file
        self.with_mask = with_mask
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

    def _get_img_pools(self):
        self.img_pools = {}
        # set mask ind dict
        if self.with_mask:
            self.mask_dict = {}
        else:
            self.mask_dict = None
        # get image pool
        for k, df_cfg in self.df_config.items() :
            if df_cfg is None:
                self.img_pools[k] = None
                if self.with_mask: self.mask_dict[k] = None
                continue
            img_pools = []
            for r_id, row in df_cfg.iterrows():
                index_files = row["index_files"]
                img_pool = _get_img_ls(index_files)
                # process mask file
                mask_index = row["mask_index"]
                if self.with_mask and mask_index is not '':
                    mask_index = [fp.strip() for fp in mask_index.split(',') if fp.strip() != ""]
                    msk_pool = _get_img_ls(mask_index)
                    if k in self.mask_dict.keys():
                        self.mask_dict[k].update(_match_img_mask(img_pool, msk_pool))
                    else:
                        self.mask_dict[k] = {}
                        self.mask_dict[k].update(_match_img_mask(img_pool, msk_pool))
                img_pools.append(img_pool)
            self.img_pools[k] = img_pools

    def get_set(self, name="train"):
        """ Get set
            return config list, image pool, mask dict
        """
        if not hasattr(self, 'img_pools'):
            self._get_img_pools()

        if self.with_mask:
            mask_dict = self.mask_dict[name]
        else:
            mask_dict = None
        return self.df_config[name].to_dict('list'), self.img_pools[name], mask_dict

    def sampling(self, name="train"):
        """ Return sampled pools
            label, image list, mpp list, mask list
        """
        config, sample_pool, mask_dict = self.get_set(name)
        label = []
        img_ls = []
        mpp_ls = []
        msk_ls = []
        grp_names = []
        for i in range(len(sample_pool)):
            mpp = float(config["subset_mpp"][i])
            nb_smp = int(config["subset_sample"][i])
            lb_smp = int(config["subset_label"][i])
            pl_smp = sample_pool[i]
            pl_msk = [None] * len(pl_smp)
            if self.with_mask and lb_smp:
                pl_tmp = []
                pl_msk = []
                # check whether the mask index exist.
                if config["mask_index"][i] is not '':
                    for smp_p in pl_smp:
                        msk_p = mask_dict[smp_p]
                        if msk_p is None:
                            logger.warning(f"{smp_p} has no matched mask.")
                            continue
                        pl_tmp.append(smp_p)
                        pl_msk.append(msk_p)
                pl_smp = pl_tmp
                nb_smp = min(len(pl_smp), nb_smp)
            # sampling
            if nb_smp:
                sample_idx = random.sample(list(range(len(pl_smp))), nb_smp)
                sample_idx = sorted(sample_idx)
                # record
                label += [lb_smp] * nb_smp
                img_ls += np.array(pl_smp)[sample_idx].tolist()
                mpp_ls += [mpp] * nb_smp
                msk_ls += np.array(pl_msk)[sample_idx].tolist()
                grp_names += [config["group_name"][i] + '_' + config["subset_name"][i]] * nb_smp
            else:
                logger.warning("no mask found in {} Group: {} Subset: {}, check "
                               "the mask index file.".format(name, config["group_name"][i], config["subset_name"][i]))
        return label, img_ls, mpp_ls, msk_ls, grp_names

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


def process_df_config(df_config: pd.DataFrame):
    """ Process single sheet of dataset config file
        *NOTE: This is a inplace operation.
    """
    # full fill merged col
    for col_ind in ["group_name", "subset_label", "subset_mpp", "group_nb"]:
        df_config[col_ind] = df_config[col_ind].ffill()
    df_config.fillna('', inplace=True)
    df_config = df_config[df_config["flg_use"] == 1]
    for r_idx, row in df_config.iterrows():
        index_files = [fp.strip() for fp in row["index_files"].split(',') if fp.strip() != ""]
        subset_total = 0
        for fp in index_files:
            with open(fp, 'r') as f:
                subset_total += len(f.readlines())
        subset_sample = max(1, int(float(row["group_nb"]) * float(row["subset_ratio"])))  # at least 1 sample
        df_config.at[r_idx, "subset_total"] = subset_total
        df_config.at[r_idx, "subset_sample"] = min(subset_sample, subset_total)
        df_config.at[r_idx, "index_files"] = index_files
    return df_config


def _get_img_ls(ind_files):
    img_ls = []
    for fp in ind_files:
        with open(fp, 'r') as f:
            img_ls += [l.strip() for l in f.readlines()]
    return img_ls


def _match_img_mask(img_pool, mask_pool):
    """ Match image mask according to image name.
        If missing, match fuzzy.
    """
    mat_dict = {}

    mask_ls = mask_pool.copy()
    mask_names = [os.path.basename(p).split('.')[0] for p in mask_ls]
    for img_p in img_pool:
        img_n = os.path.basename(img_p).split('.')[0]
        try:
            # msk_n same as img_n
            msk_p = mask_ls[mask_names.index(img_n)]
            msk_n = img_n
        except ValueError as e:
            # msk_n is different from img_n
            # init
            msk_p = None
            msk_n = None
            # todo: fuzzy search
        finally:
            # always record the result. None for missing.
            mat_dict[img_p] = msk_p
        # if matched, remove matched
        if msk_p is not None:
            mask_names.remove(msk_n)
            mask_ls.remove(msk_p)
    return mat_dict
