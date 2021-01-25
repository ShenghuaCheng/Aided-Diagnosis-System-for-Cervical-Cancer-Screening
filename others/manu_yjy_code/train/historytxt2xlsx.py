# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

def hitorytxt2xlsx(pathTxt, pathSavexlsx):
    txt = open(pathTxt, "r").readlines()
    # 依次剥离四个值
    str1 = "{'val_loss': ["
    str2 = "], 'val_binary_accuracy': ["
    str3 = "], 'loss': ["
    str4 = "], 'binary_accuracy': ["
    str5 = "]}"
    val_loss = []
    val_binary_accuracy = []
    loss = []
    binary_accuracy = []
    epoch_name = []
    for i in range(np.int(len(txt)/2)):
        strlist = txt[2*i+1].split(str1)
        val_loss += [np.float64(strlist[1].split(str2)[0])]

        strlist = txt[2*i+1].split(str2)
        val_binary_accuracy += [np.float64(strlist[1].split(str3)[0])]

        strlist = txt[2*i+1].split(str3)
        loss += [np.float64(strlist[1].split(str4)[0])]

        strlist = txt[2*i+1].split(str4)
        binary_accuracy += [np.float64(strlist[1].split(str5)[0])]

        epoch_name.append(txt[2*i].split("\n")[0][5:])

    val_loss = np.array(val_loss)
    val_binary_accuracy = np.array(val_binary_accuracy)
    loss = np.array(loss)
    binary_accuracy = np.array(binary_accuracy)
    epoch_name = np.array(epoch_name)

    record_col = np.hstack(( val_loss.reshape(val_loss.shape[0],1),loss.reshape(loss.shape[0],1),
                            val_binary_accuracy.reshape(val_binary_accuracy.shape[0],1),
                            binary_accuracy.reshape(binary_accuracy.shape[0],1)))
    # 写入excel
    record_df = pd.DataFrame(record_col)
    # change the index and column name
    record_df.index = epoch_name
    record_df.columns = ['val_loss','loss','val_binary_accuracy','binary_accuracy']
    writer = pd.ExcelWriter(pathSavexlsx)
    record_df.to_excel(writer,'Sheet1', float_format='%.5f')  # float_format 控制精度
    writer.save()

    return None

if __name__ == '__main__':
    pathTxt = r'H:\weights\w_sdpc\model2\stage_new_class\logs\hist.txt'
    pathSave = r'H:\weights\w_sdpc\model2\stage_new_class'
    nameXlsx = '2.xlsx'
    pathSavexlsx = pathSave +'/'+ nameXlsx
    if not os.path.exists(pathSave):
        os.makedirs(pathSave)
    hitorytxt2xlsx(pathTxt, pathSavexlsx)
