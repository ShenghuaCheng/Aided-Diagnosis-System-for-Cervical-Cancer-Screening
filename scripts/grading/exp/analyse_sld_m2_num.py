# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: analyse_sld_m2_num.py
@Date: 2019/12/23 
@Time: 18:51
@Desc:
'''
import os
from utils.grading.dataset import WSIDataSet

if __name__ == '__main__':
    read_in_dicts = {
        '': [
            # r'Shengfuyou_1th\Positive',
            # r'Shengfuyou_1th\Negative',
            # r'Shengfuyou_2th\Positive',
            # r'Shengfuyou_2th\Negative',
            # r'3D_Shengfuyou_3th\Positive',
            # r'3D_Shengfuyou_3th\Negative',
        ],
        'our': [
            # r'Positive\Shengfuyou_3th',
            # r'Negative\ShengFY-N-L240(origin date)',
            # r'Positive\Shengfuyou_4th',
            # r'Positive\Shengfuyou_5th\svs-20',
            r'Positive\Tongji_4th',
            r'Negative\Tongji_4th_neg',
            r'Positive\Tongji_5th',
        ],
        'SZSQ_originaldata': [
            # r'Shengfuyou_1th',
            # r'Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X',
            # r'Shengfuyou_3th\negative\Shengfuyou_3th_negative_40X',
            # r'Shengfuyou_5th\positive\Shengfuyou_5th_positive_40X',
            # r'Shengfuyou_6th\Shengfuyou_6th_negtive_40X',
            # r'Shengfuyou_7th\positive\Shengfuyou_7th_positive_40x',
            # r'Shengfuyou_7th\negative\Shengfuyou_7th_negative_40x',
            # r'Shengfuyou_8th\positive\pos_ascus',
            # r'Shengfuyou_8th\positive\pos_hsil',
            # r'Shengfuyou_8th\positive\pos_lsil',
            # r'Shengfuyou_8th\negative',

            r'Tongji_3th\positive\tongji_3th_positive_40x',
            r'Tongji_3th\negative\tongji_3th_negtive_40x',
            r'Tongji_4th\positive',
            r'Tongji_4th\negative',
            r'Tongji_5th\tongji_5th_positive\tongji_5th_positive_7us',
            r'Tongji_5th\tongji_5th_negative\tongji_5th_negative_7us',
            r'Tongji_6th\positive',
            r'Tongji_6th\negative',
            r'Tongji_7th\positive',
            r'Tongji_7th\negative',
            r'Tongji_8th\positive',
            r'Tongji_8th\negative',
            r'Tongji_9th\positive',
            r'Tongji_9th\negative',

            # r'XiaoYuwei\positive',
            # r'XiaoYuwei\negative',
            # r'XiaoYuwei2\positive',
            # r'XiaoYuwei2\negative',
        ],
        'SrpData': [
            # r'out\xyw1\positive',
            # r'out\xyw1\negative',
            # r'out\xyw2\positive',
            # r'out\xyw2\negative',
            # r'out\xyw3\positive',
            # r'out\xyw3\negative',
            #
            # r'out_csh\xyw1\positive',
            # r'out_csh\xyw1\negative',
            # r'out_csh\xyw2\positive',
            # r'out_csh\xyw2\negative',
            # r'out_csh\xyw3\positive',
            # r'out_csh\xyw3\negative',
            #
            # r'out2\xyw1\positive',
            # r'out2\xyw1\negative',
            # r'out2\xyw2\positive',
            # r'out2\xyw2\negative',
            # r'out2\xyw3\positive',
            # r'out2\xyw3\negative',
            #
            # r'out2_bak\xyw1\positive',
            # r'out2_bak\xyw1\negative',
            # r'out2_bak\xyw2\positive',
            # r'out2_bak\xyw2\negative',
            # r'out2_bak\xyw3\positive',
            # r'out2_bak\xyw3\negative',
            #
            # r'out2_bak2\xyw1\positive',
            # r'out2_bak2\xyw1\negative',
            # r'out2_bak2\xyw2\positive',
            # r'out2_bak2\xyw2\negative',
            # r'out2_bak2\xyw3\positive',
            # r'out2_bak2\xyw3\negative',
            #
            # r'out-2019-11-17\xyw1\positive',
            # r'out-2019-11-17\xyw1\negative',
            # r'out-2019-11-17\xyw2\positive',
            # r'out-2019-11-17\xyw2\negative',
            # r'out-2019-11-17\xyw3\positive',
            # r'out-2019-11-17\xyw3\negative',
            #
            # r'out-2019-11-19\xyw1\positive',
            # r'out-2019-11-19\xyw1\negative',
            # r'out-2019-11-19\xyw2\positive',
            # r'out-2019-11-19\xyw2\negative',
            # r'out-2019-11-19\xyw3\positive',
            # r'out-2019-11-19\xyw3\negative',
        ],
    }
    slds_root = r'H:\TCTDATA'
    preds_root = r'H:\fql\rnnResult\rnn1000'
    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)

    origin_root = r'J:\liusibo\DataBase\RnnData\top100_nothres_384\origin'
    gamma_root = r'J:\liusibo\DataBase\RnnData\top100_nothres_384\gamma'

    # 问题切片，区域小于100
    file_dir = r'F:\LiuSibo\Exps\191223_rnn_explore\rnnbase\less_than_100.txt'
    with open(file_dir, 'a') as f:
        for item in wsi_data:
            name = item[1].slide_dir.lstrip(slds_root)
            origin_fld = os.path.join(origin_root, name)
            gamma_fld = os.path.join(gamma_root, name)
            if len(os.listdir(origin_fld)) < 100:
                f.write('%d,%d,%s\n' % (item[0], len(os.listdir(origin_fld)), origin_fld))
            if len(os.listdir(gamma_fld)) < 100:
                f.write('%d,%d,%s\n' % (item[0], len(os.listdir(origin_fld)), gamma_fld))

    # # 进入model2tile小于100的
    # file_dir = r'F:\LiuSibo\Exps\191223_rnn_explore\rulebase\less_than_100.txt'
    # with open(file_dir, 'a') as f:
    #     for item in wsi_data:
    #         if len(item[1].scores) < 100:
    #             f.write('%d,%d,%s\n' % (item[0], len(item[1].scores), item[1].reader.slide_dir))

