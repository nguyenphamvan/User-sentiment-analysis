import os
import json
import random

def filter_data(filename1):
    with open(filename1, 'r') as file:
        contents = json.load(file)
        file.close()

    all_data = []
    for content in contents:
        if content['comment'] == None or content['rating'] == None:
            contents.remove(content)
            continue
        data = {}
        if float(content['rating']) < 4.5:
            data['comment'] = content['comment']
            data['label'] = 1
        elif float(content['rating']) > 7.5:
            data['comment'] = content['comment']
            data['label'] = 0
        else:
            continue
        all_data.append(data)

    return all_data

def load_data(pos_data_path, neg_data_path):
    list_file_pos = os.listdir(pos_data_path)
    list_file_neg = os.listdir(neg_data_path)
    data_pos = []
    data_neg = []

    for file in list_file_pos:
        data = {}
        with open(os.path.join(pos_data_path, file), 'r', encoding='utf-8') as f:
            data['comment'] = f.read()
            data['label'] = 0
            data_pos.append(data)
    f.close()

    for file in list_file_neg:
        data = {}
        with open(os.path.join(neg_data_path, file), 'r', encoding='utf-8') as f1:
            data['comment'] = f1.read()
            data['label'] = 0
            data_neg.append(data)
    f1.close()
    return data_pos, data_neg

def load_data_format(filename1):
    with open(filename1, 'r') as file:
        contents = json.load(file)
        file.close()
    return contents

def combine_data_crawl(folder_path):
    list_file = os.listdir(folder_path)
    negative = []
    positive = []
    for file in list_file:
        data = filter_data('data/comment_data.json')
        negative += [neg for neg in data if neg['label'] == 1]
        positive += [pos for pos in data if pos['label'] == 0]

    positive = positive[:len(negative)]
    all_data = positive + negative

    with open('data/test_data.json', 'w') as out_file:
        json.dump(all_data, out_file, ensure_ascii=False)

    print('finished dump file !!!')


# data1 = filter_data('data/comment_data.json')
# nag_cmt1 = [nag for nag in data1 if nag['label']==1]
# data2 = filter_data('data/comment_data_large.json')
# nag_cmt2 = [nag for nag in data2 if nag['label']==1]
# data3 = filter_data('data/foody_can_tho.json')
# nag_cmt3 = [nag for nag in data3 if nag['label']==1]
# data4 = filter_data('data/foody_da_nang.json')
# nag_cmt4 = [nag for nag in data4 if nag['label']==1]
# data5 = filter_data('data/hai-phong.json')
# nag_cmt5 = [nag for nag in data5 if nag['label']==1]
#
# data6 = filter_data('data/foodycomment.json')
# nag_cmt6 = [nag for nag in data6 if nag['label']==1]
# data7 = filter_data('data/foodycomment_1.json')
# nag_cmt7 = [nag for nag in data7 if nag['label']==1]
# data8 = filter_data('data/foodycomment_hcm.json')
# nag_cmt8 = [nag for nag in data8 if nag['label']==1]
# data9 = filter_data('data/hcm_new.json')
# nag_cmt9 = [nag for nag in data9 if nag['label']==1]
#
# pos_cmt8 = [nag for nag in data8 if nag['label']==0]
# pos_cmt7 = [nag for nag in data7 if nag['label']==0]
# pos_cmt6 = [nag for nag in data6 if nag['label']==0]
# pos_cmt5 = [nag for nag in data5 if nag['label']==0]
#
# print(len(pos_cmt8 + pos_cmt7 + pos_cmt6 + pos_cmt5))
#
# pos_cmt = pos_cmt8 + pos_cmt7 + pos_cmt6 + pos_cmt5
# nag_cmt = nag_cmt1 + nag_cmt2 + nag_cmt3 + nag_cmt4 + nag_cmt5 + nag_cmt6 + nag_cmt7 + nag_cmt8 + nag_cmt9
# print(len(nag_cmt))
#
# new_data = pos_cmt[:8000] + nag_cmt[:8000]
# print(type(new_data))
# print(new_data[1])
# random.shuffle(new_data)
# print(new_data[1])
#
# with open('test_data.json','w') as out_file:
#     json.dump(new_data, out_file, ensure_ascii=False)
#
# data_pos, data_neg = load_data('data/train/pos', 'data/train/neg')

