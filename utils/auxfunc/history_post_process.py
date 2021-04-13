# -*- coding:utf-8 -*-
import os
import collections
import numpy as np
import pandas as pd


def parse_hist_txt(file_dir, matrix=None):
    if matrix is None:
        matrix = ["val_loss", "val_binary_accuracy", "loss", "binary_accuracy"]
    with open(file_dir, 'r') as f:
        hist_total = f.readlines()
    block = []
    acc_record = []
    hist_dict = collections.OrderedDict()
    for idx in range(len(hist_total)//2):
        k = hist_total[idx*2].strip("\n")
        if "Block" not in k:
            raise ValueError("Check format of hist file, wrong line: line %d" % idx*2)
        v = hist_total[idx*2+1]
        v_dict = {}
        for i, v_key in enumerate(matrix):
            v_value = float(v.split(']')[i].split('[')[-1])
            v_dict[v_key] = v_value

        hist_dict[k] = v_dict
        block.append(k)
        acc_record.append(v_dict)
    return hist_dict, block, acc_record


if __name__ == '__main__':
    file_dir = r"hist.txt"
    save_root = r'\weights'
    save_val = os.path.join(save_root, 'val_record.xlsx')
    save_test = os.path.join(save_root, 'test_record.xlsx')
    hist_dict, block, acc_record = parse_hist_txt(file_dir)
    val_record = []
    val_name = []
    test_record = []
    test_name = []
    for idx in range(len(block)):
        if int(block[idx].strip('Block'))%2==0:
            test_name.append(block[idx])
            test_record.append(acc_record[idx])
        else:
            val_name.append(block[idx])
            val_record.append(acc_record[idx])
    val_df = pd.DataFrame.from_records(val_record, index=val_name)
    test_df = pd.DataFrame.from_records(test_record, index=test_name)

    wt = pd.ExcelWriter(save_val)
    val_df.to_excel(wt)
    wt.close()
    wt = pd.ExcelWriter(save_test)
    test_df.to_excel(wt)
    wt.close()
