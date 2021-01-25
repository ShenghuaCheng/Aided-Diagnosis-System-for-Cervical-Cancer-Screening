# -*- coding:utf-8 -*-
'''
@Author: LiuSibo
@Project: Screening_legacy
@File: wsi_dataset.py
@Date: 2019/7/25 
@Time: 17:03
@Desc:
'''
import os
import sys
sys.path.append(r'F:\LiuSibo\Codes\Projects\Screening_legacy\utils')
sys.path.append(r'F:\LiuSibo\Codes\Projects\Screening_legacy\bin\whole_slide_rank')
import time
import json

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import exposure

from utils.sldreader import SlideReader


def gamma(img):
    """输入为bgr"""
    img_gamma = exposure.adjust_gamma(img, 0.6)
    return img_gamma


class SlideObj():
    def __init__(self, img_fld, pred_fld, sld_name, result_type, rnn_sample_size=256):
        self.img_fld = img_fld
        self.pred_fld = pred_fld
        self.sld_name = sld_name
        self.result_type = result_type
        self.rnn_sample_size = rnn_sample_size

        self._sld_postfix = self.sld_name.split('.')[-1]
        self._verify_res()
        if result_type == 'npy':
            self._parse_npy()
        elif result_type == 'json':
            self._parse_json()
        else:
            raise ValueError('arg result_type should be \'json\' or \'npy\' ')

    def _verify_res(self):
        if 'our' in self.img_fld:
            self._res = 0.293
        if 'SZSQ_originaldata' in self.img_fld or 'szsq' in self.img_fld:
            self._res = 0.180

    def _parse_json_legacy(self):
        """ 未进入model2的取model1中心
        """
        result_dir = os.path.join(self.pred_fld, 'all_results_json', self.sld_name.rstrip(self._sld_postfix) + self.result_type)
        f = open(result_dir)
        all_result = json.load(f)
        self.score_1 = np.array(all_result['Score1'])
        self.score_12 = np.array(all_result['Score12'])
        self.abs_model1_regions = np.stack(all_result['Model1Block'])
        abs_model12_regions = []
        for idx in range(len(self.abs_model1_regions)):
            if len(all_result['Score2'][idx]):
                max_idx = np.argmax(all_result['Score2'][idx])
                model2_tl = np.array(all_result['Model2Anchor'][idx][max_idx]) - np.array([int((self.rnn_sample_size/2) * 0.293 / self._res),
                                                                                           int((self.rnn_sample_size/2) * 0.293 / self._res)])
                model2_tl = np.clip(model2_tl, 0, None)
                abs_model12_regions.append(model2_tl)
            else:
                abs_model12_regions.append(np.array(all_result['Model1Block'][idx]) +
                                           np.array([int((512 - self.rnn_sample_size/2) / 2 * 0.586 / self._res),
                                                     int((512 - self.rnn_sample_size / 2) / 2 * 0.586 / self._res)]))
        self.abs_model12_regions = np.stack(abs_model12_regions)
        f.close()

    def _parse_json(self):
        """ 未进入model2的取model1随意定位（第一个）
        """

        result_dir = os.path.join(self.pred_fld, 'all_results_json', self.sld_name.rstrip(self._sld_postfix) + self.result_type)
        f = open(result_dir)
        all_result = json.load(f)
        self.score_1 = np.array(all_result['Score1'])
        self.score_12 = np.array(all_result['Score12'])
        self.abs_model1_regions = np.stack(all_result['Model1Block'])
        abs_model12_regions = []
        for idx in range(len(self.abs_model1_regions)):
            if len(all_result['Score2'][idx]):
                max_idx = np.argmax(all_result['Score2'][idx])
                model2_tl = np.array(all_result['Model2Anchor'][idx][max_idx]) - np.array([int((self.rnn_sample_size/2) * 0.293 / self._res),
                                                                                           int((self.rnn_sample_size/2) * 0.293 / self._res)])
                model2_tl = np.clip(model2_tl, 0, None)
                abs_model12_regions.append(model2_tl)
            else:
                abs_model12_regions.append(np.clip(np.array(all_result['Model2Anchor'][idx][0]) +
                                           - np.array([int((self.rnn_sample_size / 2) * 0.293 / self._res),
                                                       int((self.rnn_sample_size / 2) * 0.293 / self._res)]), 0, None))
        self.abs_model12_regions = np.stack(abs_model12_regions)
        f.close()

    def _parse_npy(self):
        subfld = 'model2'
        sld_name = self.sld_name.rstrip('.' + self._sld_postfix)
        fld = self.pred_fld

        score_12 = np.load(os.path.join(fld, subfld, sld_name + '_p12.npy')).ravel()
        score_2 = np.load(os.path.join(fld, subfld, sld_name + '_p2.npy')).ravel()
        model2_region_tl = np.load(os.path.join(fld, subfld, sld_name + '_s2.npy'))
        model2_dict = np.load(os.path.join(fld, subfld, sld_name + '_dictionary.npy'))

        score_1 = np.load(os.path.join(fld, sld_name + '_p.npy')).ravel()
        fov_tl = np.load(os.path.join(fld, sld_name + '_s.npy'))
        rel_model1_region_tl = np.load(os.path.join(fld, 'startpointlist_split.npy'))

        abs_model1_regions = []
        abs_model12_regions = []
        for idx in range(len(model2_dict)):
            nb_response = model2_dict[idx]
            abs_m1_region_tl = fov_tl[idx // 15] + (rel_model1_region_tl[idx % 15] * 0.586 / self._res).astype(int)

            if nb_response:
                m2_pred_scores = score_2[np.sum(model2_dict[:idx]): np.sum(model2_dict[:idx]) + nb_response]
                m2_reg_tls = model2_region_tl[np.sum(model2_dict[:idx]): np.sum(model2_dict[:idx]) + nb_response]

                max_m2_pred_scores_idx = np.argmax(m2_pred_scores)
                abs_max_m2_reg_tl = (m2_reg_tls[max_m2_pred_scores_idx] * 0.586 / self._res).astype(int) + fov_tl[idx // 15]
                # max_m2_pred_score = m2_pred_scores[max_m2_pred_scores_idx]  # 用于验证
            else:
                # max_m2_pred_score = score_1[idx]  # 用于验证
                abs_max_m2_reg_tl = abs_m1_region_tl + [int((512 * 3 / 8) * 0.586 / self._res),
                                                        int((512 * 3 / 8) * 0.586 / self._res)]  # 大块转取中心
            for i, v in enumerate(abs_max_m2_reg_tl):
                if v < 0:
                    abs_max_m2_reg_tl[i] = 0
            abs_model1_regions.append(abs_m1_region_tl)
            abs_model12_regions.append(abs_max_m2_reg_tl)

        self.score_1 = score_1
        self.score_12 = score_12
        self.abs_model1_regions = np.stack(abs_model1_regions)
        self.abs_model12_regions = np.stack(abs_model12_regions)

    def create_img_data(self, topk, mink=10, threshold=0, save_root=None, gamma_root=None):
        """返回原始分辨率bgr图片组
        :param topk: 取排名前几
        :param size: 0.293下大小
        :return:  [topk, h, w, c] bgr
        """
        idx_sorted = np.argsort(self.score_12)[::-1]
        sld_reader = SlideReader(os.path.join(self.img_fld, self.sld_name))
        img_data = []
        for i in range(topk):
            idx_crop_tl = idx_sorted[i]
            if i > mink:
                if self.score_12[idx_crop_tl] < threshold:
                    break
            crop_tl = self.abs_model12_regions[idx_crop_tl]
            img = sld_reader.get_tile(crop_tl, (int(self.rnn_sample_size*0.293/self._res), int(self.rnn_sample_size*0.293/self._res)))
            img = img[..., ::-1]
            img_data.append(img)
            if save_root:
                save_fld = os.path.join(save_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
                if not os.path.exists(save_fld):
                    print('create %s' % save_fld)
                    os.makedirs(save_fld)
                cv2.imwrite(os.path.join(save_fld, '%d.tif' % i), img)

                if gamma_root:
                    gamma_save_fld = os.path.join(gamma_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
                    if not os.path.exists(gamma_save_fld):
                        print('create %s' % save_fld)
                        os.makedirs(gamma_save_fld)
                    cv2.imwrite(os.path.join(gamma_save_fld, '%d.tif' % i), exposure.adjust_gamma(img, 0.6))

        return np.stack(img_data)

    def create_unique_data(self, topk, mink=10, threshold=0, save_root=None, gamma_root=None):
        """返回原始分辨率bgr图片组
        :param topk: 取排名前几
        :param size: 0.293下大小
        :return:  [topk, h, w, c] bgr
        """
        idx_sorted = np.argsort(self.score_12)[::-1]
        sld_reader = SlideReader(os.path.join(self.img_fld, self.sld_name))
        img_data = []
        top_tl_record = []
        for i in range(len(idx_sorted)):
            idx_crop_tl = idx_sorted[i]
            crop_tl = self.abs_model12_regions[idx_crop_tl]
            # 去重
            repeat = False
            for tl in top_tl_record:
                if (tl[0] - int(self.rnn_sample_size*0.293*3/self._res/4)) < crop_tl[0] < (tl[0] + int(self.rnn_sample_size*0.293*3/self._res/4)) or \
                    (tl[1] - int(self.rnn_sample_size*0.293*3/self._res/4)) < crop_tl[1] < (tl[1] + int(self.rnn_sample_size*0.293*3/self._res/4)):
                    repeat = True
                    break
            if repeat:
                continue
            top_tl_record.append(crop_tl)

            if len(top_tl_record) > mink:
                if self.score_12[idx_crop_tl] < threshold:
                    break
            if len(top_tl_record) > topk:
                break

            img = sld_reader.get_tile(crop_tl, (int(self.rnn_sample_size*0.293/self._res), int(self.rnn_sample_size*0.293/self._res)))
            img = img[..., ::-1]
            img_data.append(img)
            if save_root:
                save_fld = os.path.join(save_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
                if not os.path.exists(save_fld):
                    print('create %s' % save_fld)
                    os.makedirs(save_fld)
                cv2.imwrite(os.path.join(save_fld, '%d.tif' % len(top_tl_record)), img)

                if gamma_root:
                    gamma_save_fld = os.path.join(gamma_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
                    if not os.path.exists(gamma_save_fld):
                        print('create %s' % save_fld)
                        os.makedirs(gamma_save_fld)
                    cv2.imwrite(os.path.join(gamma_save_fld, '%d.tif' % len(top_tl_record)), exposure.adjust_gamma(img, 0.6))
        return np.stack(img_data)

    def create_synthetic_img(self, synthetic_recom_root, save_root=None, gamma_root=None):
        """返回原始分辨率bgr图片组
        :param topk: 取排名前几
        :param size: 0.293下大小
        :return:  [topk, h, w, c] bgr
        """
        sld_reader = SlideReader(os.path.join(self.img_fld, self.sld_name))
        img_root = os.path.join(synthetic_recom_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
        if not os.path.exists(img_root):
            img_root = os.path.join(synthetic_recom_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name.rstrip('.' + self._sld_postfix))
        name_list = os.listdir(img_root)
        for name in name_list:
            _, _, slide_id, order, x, y, metric, *description = name.rstrip('.tif').split('_')
            r_size = (int(self.rnn_sample_size*0.293/self._res), int(self.rnn_sample_size*0.293/self._res))
            coordination = (int(x), int(y))
            coordination_tl = (int(x)-int(r_size[0]/2), int(y)-int(r_size[1]/2))
            if 'score2' in description:
                i_0 = np.where(self.abs_model1_regions[:, 0] == coordination[0])
                i_1 = np.where(self.abs_model1_regions[:, 1] == coordination[1])
                idx = np.intersect1d(i_0, i_1)[0]
                coordination_tl = self.abs_model12_regions[idx]
            img = sld_reader.get_tile(coordination_tl, r_size)
            img = img[..., ::-1]
            if save_root:
                save_fld = os.path.join(save_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
                if not os.path.exists(save_fld):
                    print('create %s' % save_fld)
                    os.makedirs(save_fld)
                cv2.imwrite(os.path.join(save_fld, '%s.tif' % order), img)

                if gamma_root:
                    gamma_save_fld = os.path.join(gamma_root, self.img_fld.lstrip('H:\\TCTDATA\\'), self.sld_name)
                    if not os.path.exists(gamma_save_fld):
                        print('create %s' % save_fld)
                        os.makedirs(gamma_save_fld)
                    cv2.imwrite(os.path.join(gamma_save_fld, '%s.tif' % order), exposure.adjust_gamma(img, 0.6))

    def create_feature_data(self, trunc=None):
        slide_score = self.score_1
        m2_score = self.score_12
        if trunc:
            slide_score, m2_score = top_n_truncate(slide_score, m2_score, trunc)

        # old model1
        max_score, avg_score, median_score, slide_std, \
        nb_thre005, nb_thre05, nb_thre08, nb_thre09, nb_thre095, \
        thre005_sum_score, thre05_sum_score, thre08_sum_score, thre09_sum_score, thre095_sum_score, \
        thre005_avg_score, thre05_avg_score, thre08_avg_score, thre09_avg_score, thre095_avg_score = get_slide_features(
            slide_score)
        # olde model2
        top_n_score, _, _, _, _, \
        _, _, _, nb_thre09, nb_thre095, \
        _, _, _, thre09_sum_score, thre095_sum_score, \
        _, _, _, thre09_avg_score, thre095_avg_score = get_slide_features_m2(m2_score)

        features = top_n_score + [avg_score, median_score, slide_std,
                                  nb_thre005, nb_thre05, nb_thre08, nb_thre09, nb_thre095,
                                  thre005_sum_score, thre05_sum_score, thre08_sum_score, thre09_sum_score,
                                  thre095_sum_score,
                                  thre005_avg_score, thre05_avg_score, thre08_avg_score, thre09_avg_score,
                                  thre095_avg_score
                                  ]
        return np.reshape(features, (-1, 1))

    def get_score_distribution(self, bins, trunc=12000, s1_flag=False, save_root=None):
        score = self.score_12
        if s1_flag:
            score = self.score_1
        if trunc:
            slide_score, m2_score = top_n_truncate(self.score_1, self.score_12, trunc)
            score = m2_score
            if s1_flag:
                score = slide_score
        hist = np.histogram(score, bins, range=(0, 1))
        h_bar, p_bar = hist[0], hist[1]
        fig = plt.figure(figsize=(20, 11))
        plt.bar(p_bar[:-1], h_bar, width=1/bins, align='center', color='b', alpha=0.6)
        x_ticks = [x for x in np.linspace(0, 1, bins+1)]
        plt.xlim((-1/bins, 1))
        plt.ylim((0, len(score)))
        plt.xticks(x_ticks, ['%.2f' % x for x in x_ticks])
        plt.legend(['%s %d' % (self.sld_name ,len(self.score_12))],
                   ncol=2, bbox_to_anchor=(0, 1),
                   loc='lower left', fontsize='small')
        for y, x in zip(list(h_bar), list(p_bar[:-1])):
            plt.text(x , y, '%d' % y, color='r', alpha=0.6, horizontalalignment='center')
        if save_root:
            plt.savefig(os.path.join(save_root, '%s.png' % self.sld_name))
        else:
            # return fig
            plt.show()

    def __repr__(self):
        attr = 'Img file dir: %s\n' % os.path.join(self.img_fld, self.sld_name)
        attr += 'Resolution: %.3f\n' % self._res
        attr += 'Pred file fld: %s\n' % self.pred_fld
        return attr

    def __del__(self):
        pass


class WSIDataSet():
    _slides_root = r'H:\TCTDATA'

    def __init__(self, img_srcs, sld_srcs, prediction_root, result_type, rnn_sample_size=256):
        self._img_srcs = img_srcs
        self._sld_srcs = sld_srcs
        self._prediction_root = prediction_root
        self._result_type = result_type
        self.rnn_sample_size = rnn_sample_size

        print('Organaizing data')
        since = time.time()
        self._organize_dataset()
        print('Consume %.2f' % (time.time()-since))

    def _organize_dataset(self):
        self.all_data = []
        for instru, sld_preps in zip(self._img_srcs, self._sld_srcs):
            for sld_prep in sld_preps:
                # img_fld = os.path.split(os.path.join(self._slides_root, instru, sld_prep))[0]
                img_fld = os.path.join(self._slides_root, instru, sld_prep)
                pred_fld = os.path.join(self._prediction_root, instru, sld_prep)
                sld_list = [f for f in os.listdir(img_fld) if '.svs' in f or '.sdpc' in f]
                label = 1
                if 'neg' in sld_prep or 'Neg' in sld_prep:
                    label = 0
                for sld_name in sld_list:
                    try:
                        sld_obj = SlideObj(img_fld, pred_fld, sld_name, self._result_type, self.rnn_sample_size)
                        self.all_data.append([sld_obj, label])
                    except:
                        print('result file missing %s' % os.path.join(pred_fld, sld_name))

    def __getitem__(self, item):
        return self.all_data[item]

    def __len__(self):
        return len(self.all_data)


if __name__ == '__main__':
    # img_srcs = [
    #     'our',
    #     'SZSQ_originaldata',
    # ]
    # sld_srcs = [
    #     [
    #         # 'Shengfuyou_3th', 'Shengfuyou_4th', r'Shengfuyou_5th\svs-20', 'ShengFY-N-L240(origin date)',
    #         # 'ShengFY-N-L240 (origin date)', 'ShengFY-P-L240 (origin date)', '2018', r'Shengfuyou_5th\svs-10'
    #     ],
    #     [
    #         # 'Shengfuyou_3th', 'Shengfuyou_5th', 'Shengfuyou_7th',
    #
    #         r'Tongji_3th\positive\tongji_3th_positive_40x',
    #         r'Tongji_3th\negative\tongji_3th_negtive_40x',
    #         r'Tongji_4th\positive',
    #         r'Tongji_4th\negative', r'Tongji_4th\tj_4th_negative_611',
    #     ],
    # ]
    # prediction_root = r'F:\recom\model1_szsq646_model2_szsq1084'
    # wsi_data = WSIDataSet(img_srcs, sld_srcs, prediction_root)
    # wsi_data[1][0].create_img_data(10, save_root=r'F:\LiuSibo\Temp\check_img')
    # f = wsi_data[0][0].create_feature_data()

    pred_fld = r'F:\WSI_InferenceResult\model1_szsq646_model2_szsq1084\SZSQ_originaldata\Tongji_4th\positive'
    img_fld = r'H:\TCTDATA\SZSQ_originaldata\Tongji_4th\positive'
    sld_name = 'tj190401606.sdpc'
    so = SlideObj(img_fld, pred_fld, sld_name, result_type='json')
    so.create_img_data(50, save_root=r'F:\LiuSibo\Temp\check_img')