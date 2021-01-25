# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: AidedDiagnosisSystem
@File: dataset.py
@Date: 2019/11/26 
@Time: 18:10
@Desc:
'''
import os
import time
import numpy as np
import cv2
from skimage import exposure

from utils.sldreader import SlideReader


def parse_txtresults(preds_dir):
    scores = []
    anchors = []
    with open(preds_dir, 'r') as f:
        for line in f.readlines():
            s, x, y = line.split(',')
            scores.append(float(s))
            anchors.append([int(x), int(y)])
    return scores, anchors


def gamma(img):
    """输入为bgr"""
    img_gamma = exposure.adjust_gamma(img, 0.6)
    return img_gamma


class SlideObj:
    def __init__(self, sld_dir, preds_dir):
        """提供切片的绝对路径和预测结果的绝对路径
        """
        self.slide_dir = sld_dir
        if not os.path.exists(preds_dir):
            raise FileNotFoundError('Result file missing %s' % preds_dir)
        self.scores, self.anchors = parse_txtresults(preds_dir)

        # 这一步是防止坏质量切片进入，根据实验结果，一张相对完整的切片不应该小于300个model2计算
        if len(self.scores) < 100:
            raise ValueError('Make sure sld is completed (300 locations should be found) %s' % sld_dir)

    def crop_imgs_data(self, nb_max, nb_min, thres, aim_size, slds_root, save_root, save_gamma=None):
        """依照预测结果裁剪数据并保存
        :param nb_max: 最大裁剪数
        :param nb_min: 最小裁剪数
        :param thres: 最大裁剪阈值
        :param aim_size: model2 输入 pixel size=0.293 下的目标尺寸
        :param save_root: 存储原图路径
        :param save_gamma: 存储gamma变换图片路径
        :return: None
        """

        self.reader = SlideReader(self.slide_dir)  # get tile func return RGB img
        if self.reader is None:
            raise ValueError('Reader creation error! %s' % self.slide_dir)
        # saving flds config
        save_fld = os.path.join(save_root, self.reader.slide_dir.lstrip(slds_root))
        os.makedirs(save_fld, exist_ok=True)
        if save_gamma:
            gamma_fld = os.path.join(save_gamma, self.reader.slide_dir.lstrip(slds_root))
            os.makedirs(gamma_fld, exist_ok=True)
        # main loop
        for i, (score, anchor) in enumerate(zip(self.scores, self.anchors)):
            if i >= nb_min and score >= thres:
                break
            if i >= nb_max:
                break
            # read in img
            crop_size = int(aim_size*0.293/self.reader.pixel_size)
            crop_tl = np.array(anchor) - [int(aim_size*0.293/self.reader.pixel_size/2), int(aim_size*0.293/self.reader.pixel_size/2)]
            crop_tl = np.clip(crop_tl, 0, None)
            img = self.reader.get_tile(crop_tl, [crop_size, crop_size])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # save img
            cv2.imwrite(os.path.join(save_fld, '%d_%.4f.tif' % (i, score)), img)
            if save_gamma:
                img = gamma(img)
                cv2.imwrite(os.path.join(gamma_fld, '%d_%.4f.tif' % (i, score)), img)
        print('save in %s, total: %d' % (save_fld, i))
        if save_gamma:
            print('save in %s, total: %d' % (gamma_fld, i))

    def __del__(self):
        pass


class WSIDataSet:
    def __init__(self, slds_root, preds_root, read_in_dicts):
        """ organize wsi data set
        :param slds_root: 切片存放根目录
        :param preds_root: 预测结果存放根目录
        :param read_in_dicts: 需要的数据集，文件结构保持一致
        """
        self.slds_root = slds_root
        self.preds_root = preds_root
        self.read_in_dicts = read_in_dicts

        print('Organaizing data')
        since = time.time()
        self._organize_dataset()
        print('Consume %.2f' % (time.time()-since))

    def _organize_dataset(self):
        """根据提供的路径组织各个成像模式和来源数据成数据集
        """
        self.all_data = []
        for slds_module in self.read_in_dicts:
            slds_srcs = self.read_in_dicts[slds_module]
            for sld_src in slds_srcs:
                slds_fld = os.path.join(self.slds_root, slds_module, sld_src)
                preds_fld = os.path.join(self.preds_root, slds_module, sld_src)
                slds_list = [f for f in os.listdir(slds_fld)
                             if '.svs' in f or '.sdpc' in f or '.srp' in f or '.mrxs' in f]
                label = 1
                if 'neg' in sld_src or 'Neg' in sld_src or 'N' in sld_src:
                    label = 0
                for sld_name in slds_list:
                    sld_dir = os.path.join(slds_fld, sld_name)
                    preds_dir = os.path.join(preds_fld, sld_name.rsplit('.', 1)[0] + '.txt')
                    try:
                        sld_obj = SlideObj(sld_dir, preds_dir)
                        self.all_data.append([label, sld_obj])
                    except Exception as e:
                        print(e)
                        continue

    def __getitem__(self, item):
        return self.all_data[item]

    def __len__(self):
        return len(self.all_data)

    def analyse_top_scores(self, nb_top):
        labels = []
        tops = []
        for wsi_item in self.all_data:
            labels.append(wsi_item[0])
            tops.append(wsi_item[1].scores[:nb_top])
        return labels, tops


if __name__ == '__main__':
    read_in_dicts = {
        'our': [
            # r'Shengfuyou_3th',
        ],
        'SZSQ_originaldata': [
            r'Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X',
        ],
        'SrpData': [
            # r'',
        ],
    }
    slds_root = r'H:\TCTDATA'
    preds_root = r'H:\fql\rnnResult\rnn1000'

    save_root = r'J:\liusibo\tmp\origin'
    save_gamma = r'J:\liusibo\tmp\gamma'

    wsi_data = WSIDataSet(slds_root, preds_root, read_in_dicts)

    # wsi_data[0][1].crop_imgs_data(100, 100, 0, 384, slds_root, save_root, save_gamma)
    wsi_data[0][1].scores[0]

    # sld_dir = r'H:\TCTDATA\SZSQ_originaldata\Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X\1135602 0893036.sdpc'
    # preds_dir = r'H:\fql\rnnResult\rnn1000\SZSQ_originaldata\Shengfuyou_3th\positive\Shengfuyou_3th_positive_40X\1135602 0893036.txt'
    # so = SlideObj(sld_dir, preds_dir)
    # ps = int(256*0.293/so.reader.pixel_size)
    # img = so.reader.get_tile(np.array(so.anchors[3])-[int(ps/2), int(ps/2)], (ps, ps))
    # plt.imshow(img)
    # since = time.time()
    # so.crop_imgs_data(100, 100, 0, 384, r'H:\TCTDATA', r'J:\liusibo\tmp\origin', r'J:\liusibo\tmp\gamma')
    # print('consum %.3f' % (time.time()-since))