# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:57:21 2021

@author: A-WIN10
"""
import os
import numpy as np
import pandas as pd

def cal_final_rnn(raw_score):
    """
    raw_score: the single rnn scores [rnn10_1, rnn10_2, rnn20_1, rnn20_2, rnn30_1, rnn30_2]
    return final score
    """
    a10 = np.mean(r_s[:2])
    a20 = np.mean(r_s[2:4])
    a30 = np.mean(r_s[4:])
    aa = np.mean([a10, a20, a30])
    amax = np.max([a10, a20, a30])
    amin = np.min([a10, a20, a30])
    astd = np.std([a10, a20, a30])
    f_s = aa if astd>0.15 else amin if aa < 0.15 else amax
    return f_s


if __name__ == '__main__':
    raw_file = r'.xlsx'
    score_file = r'.xlsx'
    
    raw_data = pd.read_excel(raw_file, header=None).to_dict('list')
    slide_names = [os.path.split(n)[-1] for n in raw_data[0]]
    labels = raw_data[1]
    scores = np.array([raw_data[i] for i in range(2,8)]).T
    f_scores = []
    for r_s in scores:
        f_s = cal_final_rnn(r_s)
        f_scores.append(f_s)
    data = pd.DataFrame(np.array([slide_names, labels, f_scores]).T)
    wrt = pd.ExcelWriter(score_file)
    data.to_excel(wrt)
    wrt.close()
    
