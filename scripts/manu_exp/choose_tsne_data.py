# -*- coding:utf-8 -*-
import os
import random
SAMPLE_ROOTS = r'F:\LiuSibo\Codes\Projects\AidedDiagnosisSystem\doc\manu_yjy\exp2\config\Itest'


# Group = A B C D E F
def sample_labels(group, label, num=1000):
    if label==1:
        files = [f for f in os.listdir(SAMPLE_ROOTS) if group + '_' in f and '_n' not in f]
    elif label==0:
        files = [f for f in os.listdir(SAMPLE_ROOTS) if group + '_' in f and '_n' in f]
    else: raise ValueError('label should be 0 or 1')

    all_dirs_ls = []
    for file in files:
        all_dirs_ls += [l.replace('I:', 'H:') for l in open(os.path.join(SAMPLE_ROOTS, file)).readlines() if os.path.exists(l.strip().replace('I:', 'H:'))]
    if group == 'E':
        all_dirs_ls = [d for d in all_dirs_ls if 'sfy5' in d or 'sfy6' in d or 'sfy7' in d or 'sfy8' in d]
    if group == 'D':
        all_dirs_ls = [d for d in all_dirs_ls if 'tongji3' in d or 'tongji4' in d]
    print('group {} {} total num: {}'.format(group, label, len(all_dirs_ls)))
    return random.sample(all_dirs_ls, num)


if __name__ == '__main__':
    save_root = r'I:\20201218_Manu_Fig4\ABCDEF_samples'
    for group in ['A', 'B', 'C', 'D', 'E', 'F']:
        for label in [0, 1]:
            dir_ls = sample_labels(group, label)
            with open(os.path.join(save_root, '{}_{}.txt'.format(group, 'P' if label else 'N')), 'w+') as f:
                f.writelines(dir_ls)