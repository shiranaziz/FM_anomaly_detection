# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.
# The code is adapted from https://github.com/yahoo/FmFM/blob/main/data/criteo/trans_criteo_dataset.py

from tqdm import tqdm
import json
import math
import numpy as np

def trans_int_feat(val):
    return int(math.ceil(math.log(val)**2+1))

def pre_parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat=True):
    fields_cnt = int_feat_cnt + cate_feat_cnt
    # splits = line.rstrip('\n').split('\t', fields_cnt+1)

    start_index = 0 if with_int_feat else int_feat_cnt
    for idx in range(start_index, fields_cnt):
        val = int(np.abs(line[idx]))
        if idx < int_feat_cnt:
            if val > 2:
                val = trans_int_feat(val)
        else:
            val = int(val, 16)

        if val not in feat_list[idx]:
            feat_list[idx][val] = 1
        else:
            feat_list[idx][val] += 1
    return

def parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat = True):
    fields_cnt = int_feat_cnt + cate_feat_cnt
    vals = []
    start_index = 0 if with_int_feat else int_feat_cnt
    for idx in range(start_index, fields_cnt):
        val = line[idx]
        if idx < int_feat_cnt:
            val = int(np.abs(line[idx]))
            if val > 2:
                val = trans_int_feat(val)
        else:
            val = int(val, 16)

        if val not in feat_list[idx]:
            vals.append(0)
        else:
            vals.append(feat_list[idx][val])
    return vals


def feature_pre_processing(data,int_feat_cnt, cate_feat_cnt, with_int_feat=True):
    thres = 8
    new_data = np.zeros_like(data)
    for i, data_after_trans in enumerate(data.transpose()):
        # out_file = open('all_data'+str(i)+'.csv', 'w')
        # feature_index = open('feature_index'+str(i), 'w')
        # feature_json = open('features.json'+str(i), 'w')
        feat_list = []
        for j in range(data_after_trans.shape[0]+1):
            feat_list.append({})
        for row_after_trans in data_after_trans.transpose():
            pre_parse_line(row_after_trans, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat)

        for lst in feat_list[:int_feat_cnt]:
            idx = 1
            for key, val in lst.items():
                lst[key] = idx
                idx += 1

        for lst in feat_list[int_feat_cnt:]:
            idx = 1
            for key, val in lst.items():
                if val < thres:
                    del lst[key]
                else:
                    lst[key] = idx
                    idx += 1

        # for idx, field in enumerate(feat_list):
        #     # feat_id = sorted(field.items(), key=lambda x:x[1])
        #     for feat, id in field.items():
        #         feature_index.write('field_%02d\1|raw_feat_%s|\1%d\n' % (idx+1, str(feat), id))
        # feature_index.close()

        for k,line in enumerate(data_after_trans.transpose()):
            vals = parse_line(line, feat_list, int_feat_cnt, cate_feat_cnt, with_int_feat)
            new_data[k,:,i] = vals

        # feature_meta = []
        # for idx in range(1, int_feat_cnt + cate_feat_cnt + 1):
        #     feature_meta.append(('field_%02d' % idx, 'CATEGORICAL', 20))
        # json.dump(feature_meta, feature_json, indent=2)
        #
        # out_file.close()

    return new_data


