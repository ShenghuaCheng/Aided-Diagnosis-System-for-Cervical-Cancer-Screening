# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: write_ini_files.py
@Date: 2020/6/15 
@Time: 15:16
@Desc:
'''

import os

if __name__ == '__main__':
    config_root = r"I:\20200615_HWToolResults\HWmodel123Rnn_tools\HWTools_configs"
    config_save = r"I:\20200615_HWToolResults\HWmodel123Rnn_tools\config"
    os.makedirs(config_save, exist_ok=True)
    config_list = os.listdir(config_root)
    m1_dir = r"I:\\20200615_HWToolResults\\pbs\\NewLabelsLtTJ_model1_236_128_model.pb"
    m2_dir = r"I:\\20200615_HWToolResults\\pbs\\NewLabelsLtTJ_model2_200_model.pb"
    save_root = r"I:\\20200615_HWToolResults\\results"
    slide_root = r"H:\\TCTDATA"
    with open(os.path.join(r"I:\20200615_HWToolResults\HWmodel123Rnn_tools\config\config.ini"), "r") as f:
        txt = f.readlines()

    for config in config_list:
        with open(os.path.join(config_root, config), "r") as f:
            contains = f.readlines()

        with open(os.path.join(config_save, config), "w") as f:
           for idx, t in enumerate(txt):
               if idx in [1, 2]:
                   f.write(contains[idx])
               else:
                   f.write(t)

