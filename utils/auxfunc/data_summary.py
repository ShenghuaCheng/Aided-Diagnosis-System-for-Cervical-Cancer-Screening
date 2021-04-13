# -*- coding:utf-8 -*-

import os
from functools import partial
import numpy as np
from utils.auxfunc.parse_labelfile import parse_csv, parse_xml


class SubSet:
    _label_list = None
    _label_pf = {0: None, 1: 'csv', 2: 'xml'}
    all_labels = None
    label_num = None

    def __init__(self, slide_fld, label_fld, label_mode):
        """指定数据的来源、标注文件夹和标注的模式，返回统计数据
        label_mode
        =========
         0 | null
         1 | csv
         2 | xml
        =========
        """
        self.slide_fld = slide_fld
        self.label_fld = label_fld
        self.label_mode = label_mode

        self._slide_list = [f for f in os.listdir(slide_fld) if '.mrxs' in f or '.svs' in f or '.sdpc' in f or
                            '.srp' in f]
        self._slide_pf = self._slide_list[0].split('.')[-1]

        if label_mode:
            self._label_list = [f for f in os.listdir(label_fld)]

        self.name_list = self._valid_names()
        if label_mode:
            self._count_labels()

    def _valid_names(self):
        if self._label_list is None:
            return self._slide_list
        sld_n = [s.split('.')[0] for s in self._slide_list]
        lab_n = [l.split('.')[0] for l in self._label_list]
        return list(np.intersect1d(sld_n, lab_n))

    def _count_labels(self):
        all_labels = []
        if self.label_mode == 1:
            parse_func = partial(parse_csv)
            name = '%s'
        elif self.label_mode == 2:
            parse_func = partial(parse_xml)
            name = '%s.' + self._label_pf[self.label_mode]

        for n in self.name_list:
            try:
                Labels, _, _, _ = parse_func(os.path.join(self.label_fld, name % n))
                all_labels += list(set(Labels))
            except:
                print('error when parsing label file: %s' %n)
                self.name_list.remove(n)
                continue

        self.all_labels = list(set(all_labels))

        counter = {}
        counter['slide'] = []
        for l in all_labels:
            counter[l] = []

        for n in self.name_list:
            counter['slide'].append(n)
            # create counter
            label_num = {}
            for l in all_labels:
                label_num[l] = 0

            # counting
            Labels, _, _, _ = parse_func(os.path.join(self.label_fld, name % n))
            for l in Labels:
                label_num[l] += 1
            for k in counter:
                if k == 'slide':
                    continue
                counter[k].append(label_num[k])

        self.label_num = label_num
        self.counter = counter


if __name__ == '__main__':
    slide_fld = r'H:\TCTDATA\Shengfuyou_1th\Positive'
    label_fld = r'H:\TCTDATA\Shengfuyou_1th\Labelfiles\csv_P'
    label_mode = 1
    # label_fld = r'H:\TCTDATA\Shengfuyou_1th\Labelfiles\XML'
    # label_mode = 2

    subset = SubSet(slide_fld, label_fld, label_mode)
