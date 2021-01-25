# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: data_args.py
@Date: 2020/5/5 
@Time: 14:53
@Desc: 数据参数的存放， 包括数据集的后缀以及存放路径，待整理
'''

class Args:
    sld_root = r"H:\TCTDATA"
    result_root = r"F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\m1m2hwresults"

    res = {"SZSQ_originaldata": 0.18, "our": 0.293, "Shengfuyou_1th": 0.243, "Shengfuyou_2th": 0.243}
    post_fix = {"SZSQ_originaldata": ".sdpc", "our": ".svs", "Shengfuyou_1th": ".mrxs", "Shengfuyou_2th": ".mrxs"}

    fld_dict = {
        "SZSQ_originaldata": [
            # r"Shengfuyou_1th",
            # r"Shengfuyou_3th\negative\Shengfuyou_3th_negative_40X",
            # r"Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X",
            r"Shengfuyou_5th\positive\Shengfuyou_5th_positive_40X",
            # r"Shengfuyou_6th\Shengfuyou_6th_negtive_40X",
            # r"Shengfuyou_7th\positive\Shengfuyou_7th_positive_40x",
            # r"Shengfuyou_7th\negative\Shengfuyou_7th_negative_40x",
            # r"Shengfuyou_8th\positive\pos_ascus",
            # r"Shengfuyou_8th\positive\pos_hsil",
            # r"Shengfuyou_8th\positive\pos_lsil",
            # r"Shengfuyou_8th\negative",
            # r"Tongji_3th\negative\tongji_3th_negtive_40x",
            # r"Tongji_3th\positive\tongji_3th_positive_40x",
            # r"Tongji_4th\negative",
            # r"Tongji_4th\positive",
            # r"Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us",
            # r"Tongji_5th\tongji_5th_negative\tongji_5th_negative_7us",
            # r"Tongji_6th\positive",
            # r"Tongji_6th\negative",
            # r"Tongji_7th\positive",
            # r"Tongji_7th\negative",
            # r"XiaoYuwei\positive",
            # r"XiaoYuwei\negative",
            # r"XiaoYuWei2\positive",
            # r"XiaoYuWei2\negative",
        ],
        "our": [
            # r"Positive\Shengfuyou_3th",
            # r"Positive\Shengfuyou_4th",
            # r"Positive\Shengfuyou_5th\svs-20",
            # r"Negative\ShengFY-N-L240(origin date)",
        ],
        "Shengfuyou_1th": [
            # "Positive",
            # "Negative",
        ],
        "Shengfuyou_2th": [
            # "Positive",
            # "Negative",
        ],
    }

    labeled = False
    circle_r = 250

    aim_res = 0.243
    aim_size = 384
    top_n = 100

    save_root = r"I:\exp_hwresults\top100"
