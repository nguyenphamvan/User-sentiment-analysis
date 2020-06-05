import os
import json

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

data = load_data_format('data/data_1.json')
print(len(data))

data_pos, data_neg = load_data('data/train/pos', 'data/train/neg')

