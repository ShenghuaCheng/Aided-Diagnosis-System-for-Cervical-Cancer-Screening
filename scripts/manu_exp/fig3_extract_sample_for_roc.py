# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: fig3_extract_sample_for_roc.py
@Date: 2021/1/2 
@Time: 14:55
@Desc: 从已经选好的消融实验固定样本当中按照比例抽取样本制作ROC曲线
'''

import os
from glob2 import glob
import numpy as np
import random
import pandas as pd
from sklearn import metrics
from utils.manu.aux_func import GROUPS, MODEL_NAMES

random.seed(42)

M2_RES_ROOT = r'I:\20201221_Manu_Fig3\results_all\m2'
M2_SAMPLE_LIST_ROOT = r'K:\liusibo\20201221_Manu_Fig3\SAMPLED_TEST_DATASET\m2'


def samples_B():
    Bset = glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'B_*.txt'))
    BPset = [d for d in Bset if 'n' not in os.path.split(d)[-1]]
    BPls = []
    for d in BPset:
        BPls += open(d, 'r').readlines()
    BP_num = len(BPls)
    BN_ndict = {
        'n': int(BP_num*390/(390+50+60)),
        'n_w': int(BP_num*50/(390+50+60)),
        'nplus': int(BP_num*60/(390+50+60))
    }
    BN_dict = {}
    for k, v in BN_ndict.items():
        BN_dict[k] = random.sample(list(range(len(open(glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'B*{}.txt'.format(k)))[0], 'r').readlines()))), v)
    return {'sfy2': BN_dict}


def samples_E():
    Eset = glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'E_*.txt'))
    EPset = [d for d in Eset if 'n' not in os.path.split(d)[-1]]

    EP_num = sum([len(open(d, 'r').readlines()) for d in EPset]) # 阳性太多，阴性样本不够采 直接取6000

    EN_ndict = {
        'sfy6': {
            'n_0': int(EP_num*440/1500),
            'n_1': int(EP_num*20/1500),
            'n_2': int(EP_num*15/1500),
            'n_5': int(EP_num*10/1500),
            'n_8': int(EP_num*9/1500),
            'n_9': int(EP_num*6/1500),
        },
        'sfy7': {
            'n_0': int(EP_num*440/1500),
            'n_1': int(EP_num*20/1500),
            'n_2': int(EP_num*15/1500),
            'n_5': int(EP_num*10/1500),
            'n_8': int(EP_num*9/1500),
            'n_9': int(EP_num*6/1500),
        },
        'sfy8': {
            'n_0': int(EP_num*440/1500),
            'n_1': int(EP_num*20/1500),
            'n_2': int(EP_num*15/1500),
            'n_5': int(EP_num*10/1500),
            'n_8': int(EP_num*9/1500),
            'n_9': int(EP_num*6/1500),
        },
    }
    EN_dict = {}
    for bat, cls in EN_ndict.items():
        EN_dict[bat] = {}
        for k, v in cls.items():
            try:
                EN_dict[bat][k] = random.sample(list(range(len(open(glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'E*{}*{}.txt'.format(bat,k)))[0], 'r').readlines()))), v)
            except ValueError:
                EN_dict[bat][k] = list(range(len(open(glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'E*{}*{}.txt'.format(bat,k)))[0], 'r').readlines())))
    return EN_dict


def samples_F():
    Fset = glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'F_*.txt'))
    FPset = [d for d in Fset if 'n' not in os.path.split(d)[-1]]

    FP_num = sum([len(open(d, 'r').readlines()) for d in FPset])
    FN_ndict = {
        'xyw1': {
            'n_0': int(FP_num*140/1500),
            'n_1': int(FP_num*20/1500),
            'n_2': int(FP_num*15/1500),
            'n_5': int(FP_num*10/1500),
            'n_8': int(FP_num*9/1500),
            'n_9': int(FP_num*6/1500),
            'n_n': int(FP_num*250/1500),
            'n_w': int(FP_num*50/1500),
        },
        'xyw2': {
            'n_0': int(FP_num*140/1500),
            'n_1': int(FP_num*20/1500),
            'n_2': int(FP_num*15/1500),
            'n_5': int(FP_num*10/1500),
            'n_8': int(FP_num*9/1500),
            'n_9': int(FP_num*6/1500),
            'n_n': int(FP_num*250/1500),
            'n_w': int(FP_num*50/1500),
        },
    }
    FN_dict= {}
    for bat, cls in FN_ndict.items():
        FN_dict[bat] = {}
        for k, v in cls.items():
            FN_dict[bat][k] = random.sample(list(range(len(open(glob(os.path.join(M2_SAMPLE_LIST_ROOT, 'F*{}*{}.txt'.format(bat, k)))[0], 'r').readlines()))), v)
    return FN_dict


if __name__ == '__main__':
    bn_dict = samples_B()
    en_dict = samples_E()
    fn_dict = samples_F()
    # save_dir_OEM = r'I:\20201221_Manu_Fig3\fig3_B_OEM_scatter.xlsx'
    # oem_wrt = pd.ExcelWriter(save_dir_OEM)
    save_dir_BH = r'I:\20201221_Manu_Fig3\fig3_EF_ALL_scatter.xlsx'
    bh_wrt = pd.ExcelWriter(save_dir_BH)

    # for mn in ['Origin', 'Enhanced', 'Mined']:
    #     labels = []
    #     scores = []
    #     res_dirs = glob(os.path.join(M2_RES_ROOT, '{}_B*.npz'.format(mn)))
    #     for res_d in res_dirs:
    #         if 'ASCUS' in res_d or 'HSIL' in res_d or 'LSIL' in res_d:
    #             scores += np.load(res_d)['score'].ravel().tolist()
    #             labels += np.load(res_d)['label'].tolist()
    #         elif 'n.npz' in res_d:
    #             k = 'n'
    #             scores += np.load(res_d)['score'][bn_dict['sfy2'][k]].ravel().tolist()
    #             labels += np.load(res_d)['label'][bn_dict['sfy2'][k]].tolist()
    #         elif 'n_w.npz' in res_d:
    #             k = 'n_w'
    #             scores += np.load(res_d)['score'][bn_dict['sfy2'][k]].ravel().tolist()
    #             labels += np.load(res_d)['label'][bn_dict['sfy2'][k]].tolist()
    #         elif 'nplus.npz' in res_d:
    #             k = 'nplus'
    #             scores += np.load(res_d)['score'][bn_dict['sfy2'][k]].ravel().tolist()
    #             labels += np.load(res_d)['label'][bn_dict['sfy2'][k]].tolist()
    #     fpr, tpr, thre = metrics.roc_curve(np.array(labels), np.array(scores))
    #     auc = metrics.roc_auc_score(np.array(labels), np.array(scores))
    #     roc = pd.DataFrame({
    #         'fpr': fpr,
    #         'tpr': tpr,
    #         'threshold': thre
    #     }).to_excel(oem_wrt, sheet_name='{}_roc'.format(mn))
    #     auc = pd.DataFrame({
    #         'auc': [auc]
    #     }).to_excel(oem_wrt, sheet_name='{}_auc'.format(mn))
    # oem_wrt.close()

    for mn in MODEL_NAMES['m2']:
        labels = []
        scores = []
        res_dirs = glob(os.path.join(M2_RES_ROOT, '{}_E*.npz'.format(mn)))
        res_dirs += glob(os.path.join(M2_RES_ROOT, '{}_F*.npz'.format(mn)))
        for res_d in res_dirs:
            if 'pos' in res_d or 'Imgs' in res_d:
                scores += np.load(res_d)['score'].ravel().tolist()
                labels += np.load(res_d)['label'].tolist()
            else:
                for post in ['n_0.npz', 'n_1.npz', 'n_2.npz', 'n_5.npz', 'n_8.npz', 'n_9.npz', 'n_n.npz', 'n_w.npz']:
                    if post in res_d:
                        for bat in ['sfy6', 'sfy7', 'sfy8', 'xyw1', 'xyw2']:
                            if bat in res_d:
                                if bat in ['sfy6', 'sfy7', 'sfy8']:
                                    scores += np.load(res_d)['score'][en_dict[bat][post.rsplit('.',1)[0]]].ravel().tolist()
                                    labels += np.load(res_d)['label'][en_dict[bat][post.rsplit('.',1)[0]]].tolist()
                                elif bat in ['xyw1', 'xyw2']:
                                    scores += np.load(res_d)['score'][fn_dict[bat][post.rsplit('.',1)[0]]].ravel().tolist()
                                    labels += np.load(res_d)['label'][fn_dict[bat][post.rsplit('.',1)[0]]].tolist()

        fpr, tpr, thre = metrics.roc_curve(np.array(labels), np.array(scores))
        auc = metrics.roc_auc_score(np.array(labels), np.array(scores))
        roc = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            'threshold': thre
        }).to_excel(bh_wrt, sheet_name='{}_roc'.format(mn))
        auc = pd.DataFrame({
            'auc': [auc]
        }).to_excel(bh_wrt, sheet_name='{}_auc'.format(mn))
    bh_wrt.close()
