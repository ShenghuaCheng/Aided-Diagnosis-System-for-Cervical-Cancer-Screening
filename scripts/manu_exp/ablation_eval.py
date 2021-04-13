# -*- coding:utf-8 -*-
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import keras.backend as K
from utils.manu.aux_func import (
    create_encoders, parse_group, get_test_data,
    IN_SHAPE, MODEL_NAMES, GROUPS
)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    res_save = r'\results_all'
    data_root = r'\SAMPLED_TEST_DATASET'

    ind_file = os.path.join(data_root, 'all_test_data.xlsx')
    # 先读数据，再逐模型
    for m_t, m_ns in MODEL_NAMES.items():  # 获取模型类别和模型列表
        feature_save = os.path.join(res_save, m_t)
        os.makedirs(feature_save, exist_ok=True)
        print("Process model type {}".format(m_t))
        in_shape = IN_SHAPE[m_t]
        writers = {item: pd.ExcelWriter(os.path.join(res_save, item+'.xlsx')) for item in m_ns}  # 逐模型创建recorder
        for grp in GROUPS:  # 获取数据组group
            print("Process Group {}".format(grp))
            records = {item: [] for item in m_ns}  # 每个group创建独立的recorder来记录
            ls_file, ls_label, ls_sample_num, ls_total_num = parse_group(grp, ind_file)

            for idx, f_dir in enumerate(ls_file):  # 逐亚类读取
                label = ls_label[idx]
                sample_n = ls_sample_num[idx]
                total_n = ls_total_num[idx]

                # 判断对于该文件是否所有模型都有结果，实现断点继续
                hist_flag = 0
                for m_n in m_ns:
                    if os.path.exists(os.path.join(feature_save, "{}_{}.npz".format(m_n, f_dir[:-4]))):
                        continue
                    else:
                        hist_flag+=1

                if hist_flag:  # 如果存在部分模型没有记录，则所有关于该文件的都需要重新预测评估
                    imgs, labs = get_test_data(os.path.join(data_root, m_t, f_dir), label, in_shape)

                for m_n in m_ns:  # 逐模型预测记录
                    # 这里的判断是为了实现断点续存
                    save_npz_dir = os.path.join(feature_save, "{}_{}.npz".format(m_n, f_dir[:-4]))
                    if hist_flag:
                        encoder, _ = create_encoders(m_n)
                        res = encoder.predict(imgs)
                        K.clear_session()  # 释放GPU

                        scores = res[0]
                        features = res[1]
                        np.savez(save_npz_dir, feature=features, score=scores, label=labs)
                    else:
                        print("Use cached record: {}".format(save_npz_dir))
                        dataz = np.load(save_npz_dir)
                        labs = dataz['label']
                        features = dataz['feature']
                        scores = dataz['score']

                    acc = metrics.accuracy_score(labs, np.round(scores))
                    records[m_n].append((f_dir, label, total_n, acc))

            print("Done Process Group {}".format(grp))
            # 每个group记录一次
            print("Recording group {}".format(grp))
            for k_m_n, rec in records.items():
                wrtr = writers[k_m_n]
                rec = pd.DataFrame(rec, columns=["txt_name", "label", "total_num", "Acc"])
                rec.to_excel(wrtr, sheet_name=grp)
            print("Done Recording group {}".format(grp))
        for _, wrtr in writers.items():
            wrtr.close()
        print("Done Process model type {}".format(m_t))




