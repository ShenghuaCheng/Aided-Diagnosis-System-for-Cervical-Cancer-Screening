# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: fig5_LSTM_test.py
@Date: 2020/12/24 
@Time: 9:16
@Desc: 该脚本用于测试基于LSTM的切片分类模型，并且计算各类数值指标，结果存至excel
'''
import os
import numpy as np
import pandas as pd
from glob2 import glob
from sklearn import metrics
from utils.manu.WSIClassify import lstm_exp
from utils.manu.aux_func import rd_imgs, create_encoders, SLIDE_TOP


def encode_slide(sld_fld, encoder):
    in_shape = (256,256,3)
    re_size = (384, 384)
    img_dirs = []
    for i in range(100):
        img_dirs += glob(os.path.join(sld_fld, "{:0>3d}_*.png".format(i)))
    if len(img_dirs)==0:
        raise RuntimeError("error at {}".format(sld_fld))
    data = rd_imgs(img_dirs, in_shape, re_size)
    res = encoder.predict(data)
    return res[0], res[1]


def encode_dataset(save_dir):
    if os.path.exists(save_dir):
        return np.load(save_dir)
    data = {
        'features': [],
        'scores': [],
        'labels': [],
        'dirs': []
    }
    encoder, _ = create_encoders('HR_model')
    for grp_n, grp_data in SLIDE_TOP.items():
        for cls, dirs in grp_data.items():
            print('encode {} {} \nfld num: {}'.format(grp_n, cls, len(dirs)))
            label = 1 if cls == 'P' else 0
            for dir in dirs:
                sld_dirs = glob(os.path.join(dir, '*'))
                print('sld num: {}'.format(len(sld_dirs)))
                for sld_dir in sld_dirs:
                    try:
                        score, feature = encode_slide(sld_dir, encoder)
                    except RuntimeError:
                        print("skip {}".format(sld_dir))
                        continue
                    data['dirs'].append(sld_dir)
                    data['labels'].append(label)
                    data['features'].append(np.stack(feature))
                    data['scores'].append(np.stack(score))
        print('done, group {}'.format(grp_n))
    data['dirs']=np.array(data['dirs'])
    data['labels']=np.array(data['labels'])
    data['features']=np.array(data['features'])
    data['scores']=np.array(data['scores'])
    np.savez(save_dir, dirs=data['dirs'], labels=data['labels'], features=data['features'], scores=data['scores'])
    return data


def create_model(top_n, hidden_units):
    return lstm_exp((top_n, 2048), hidden_units)


def clf_statistic(scores, labels):
    ststc = {
        'auc': metrics.roc_auc_score(labels, scores),
        'clf_rep': metrics.classification_report(labels, np.round(scores).astype(int), target_names=['neg', 'pos'], output_dict=True),
        'ap': metrics.average_precision_score(labels, scores)
    }
    return ststc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    # save
    save_xlsx = r'I:\20201224_Manu_Fig5_LSTM\lstm_100_256.xlsx'
    save_npz = r'I:\20201224_Manu_Fig5_LSTM\lstm_100_256_features.npz'
    # lstm setting
    top_n = 100
    hidden_units = 256
    # load test data
    test_data = encode_dataset(save_npz)
    stats = {}
    data={}
    data['dirs'] = test_data['dirs']
    data['labels'] = test_data['labels']
    # create lstm model
    lstm_model = create_model(top_n, hidden_units)
    # load weights
    weights = glob(r'I:\20201224_Manu_Fig5_LSTM\LSTM_weights\100\*.h5')
    for w in weights:
        lstm_model.load_weights(w)
        origin_preds = np.stack(lstm_model.predict(test_data['features'])).ravel().tolist()

        data['%d_origin_' % top_n + os.path.split(w)[-1]] = origin_preds

        # calculate statics
        origin_stats = clf_statistic(origin_preds, test_data['labels'])

        ocr = origin_stats['clf_rep']
        stats['%d_origin_' % top_n + os.path.split(w)[-1]] = [
            origin_stats['auc'],
            origin_stats['ap'],
            ocr['accuracy'],
            ocr['pos']['precision'],
            ocr['pos']['recall'],
            ocr['pos']['f1-score'],
            ocr['pos']['support'],
            ocr['neg']['precision'],
            ocr['neg']['recall'],
            ocr['neg']['f1-score'],
            ocr['neg']['support'],
        ]
    data_df = pd.DataFrame(data)
    stats_df = pd.DataFrame(stats, index=['auc', 'ap', 'acc', 'pprecision', 'precall', 'pf1', 'pnum',  'nprecision', 'nrecall', 'nf1', 'nnum'])
    writer = pd.ExcelWriter(save_xlsx)
    data_df.to_excel(writer, sheet_name='preds')
    stats_df.to_excel(writer, sheet_name='statics')
    writer.close()
