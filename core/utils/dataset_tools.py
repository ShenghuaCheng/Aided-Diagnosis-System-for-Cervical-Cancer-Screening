# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: dataset_tools
Description: useful tools for create dataset objects.
"""

import os
import pandas as pd

__all__ = [
    "process_df_config",
]


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
