# -*- coding: utf-8 -*-

import pandas as pd
from pyvi import ViTokenizer
import re
import string
import codecs
import json
from string import punctuation

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.externals import joblib

# Từ điển tích cực, tiêu cực, phủ định
path_nag = 'sentiment_dicts/nag.txt'
path_pos = 'sentiment_dicts/pos.txt'
path_not = 'sentiment_dicts/not.txt'

with codecs.open(path_nag, 'r', encoding='UTF-8') as f:
    nag = f.readlines()
nag_list = [n.replace('\n', '').replace(' ', '_') for n in nag]

with codecs.open(path_pos, 'r', encoding='UTF-8') as f:
    pos = f.readlines()
pos_list = [n.replace('\n', '').replace(' ', '_') for n in pos]
with codecs.open(path_not, 'r', encoding='UTF-8') as f:
    not_ = f.readlines()
not_list = [n.replace('\n', '').replace(' ', '_') for n in not_]

negative_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{', ':((', ':(((', '🤬', '🚫'}

positive_emoticons = {'=))', 'v', ';)','))' , '^^', '<3', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)', ':)))', '😆', '😂', '👻','😸','🧡','🧚','☀','🤩','🤣',
                      '🖤','💯','🤗','🙂','😝','😄','️🆗️','💟','😜','😛','😇', '😃','🤪','💪','🎉'}


def remove_punctuation(_string):
    """xóa các punctuation ra khỏi từ và câu"""
    # nếu là ký tự đặc biệt thì xóa
    for token in _string.split(' '):
        if token in punctuation:
            _string = _string.replace(token, '')
    return _string


def normalizeString(string):
    """Tách dấu ra khỏi từ"""
    s = string.lower()
    # Tách dấu câu nếu kí tự liền nhau
    marks = '[.!?,-${}()]'
    r = "([" + "\\".join(marks) + "])"
    s = re.sub(r, r" \1 ", s)
    # Thay thế nhiều spaces bằng 1 space.
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def emoj_to_string(_string):
    """Chuyênr các biểu tượng cảm xúc về text"""
    for token in _string.split(' '):
        if token in positive_emoticons:
            _string = _string.replace(token, 'positive')
        if token in negative_emoticons:
            _string = _string.replace(token, 'nagative')
    return _string


def remove_duplicate_characters(string):
    """Remove các ký tự kéo dài: vd: đẹppppppp"""
    string = re.sub(r'([A-Z])\1+', lambda m: m.group(1), string, flags=re.IGNORECASE)
    return string


def replace_abbreviations(text):
    """Chuẩn hóa 1 số sentiment words/English words"""
    text = remove_duplicate_characters(text)

    replace_list = {
        ' ô kêi ': ' ok ', ' okie ': ' ok ', ' o kê ': ' ok ',
        ' okey ': ' ok ', ' ôkê ': ' ok ', ' oki ': ' ok ', ' oke ': ' ok ', ' okay ': ' ok ', ' okê ': ' ok ',
        ' tks ': u' cám ơn ', ' thks ': u' cám ơn ', ' thanks ': u' cám ơn ', ' ths ': u' cám ơn ',
        ' thank ': u' cám ơn ',
        ' not ': u' không ', u' kg ': u' không ', ' kh ': u' không ', ' kp ': u' không phải ', u' kô ': u' không ',
        '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', ' khong ': u' không ', u' hok ': u' không ',
        ' he he ': ' positive ', ' hehe ': ' positive ', ' hihi ': ' positive ', ' haha ': ' positive ',
        ' hjhj ': ' positive ',
        ' lol ': ' nagative ', ' cc ': ' nagative ', ' cute ': u' dễ thương ', ' huhu ': ' nagative ', ' vs ': u' với ',
        ' wa ': ' quá ', ' wá ': u' quá ', ' j ': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', ' size ': u' cỡ ', u' đx ': u' được ', ' dc ': u' được ', ' đk ': u' được ',
        ' đc ': u' được ', ' authentic ': u' chuẩn chính hãng ', u' aut ': u' chuẩn chính hãng ',
        u' auth ': u' chuẩn chính hãng ', ' thick ': u' positive ', ' store ': u' cửa hàng ',
        ' shop ': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ', ' god ': u' tốt ', ' wel done ': ' tốt ',
        ' good ': u' tốt ', ' gút ': u' tốt ',
        ' sấu ': u' xấu ', ' gut ': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', ' perfect ': ' rất tốt ',
        'bt': u' bình thường ',
        ' time ': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        ' ê ̉': 'ể', 'product': 'sản phẩm', ' quality ': ' chất lượng ', ' chat ': ' chất ', ' excelent ': ' hoàn hảo ',
        ' bad ': ' tệ ', ' fresh ': ' tươi ', ' sad ': ' tệ ',
        ' date ': u' hạn sử dụng ', ' hsd ': u' hạn sử dụng ', ' quickly ': u' nhanh ', ' quick ': u' nhanh ',
        ' fast ': u' nhanh ', ' delivery ': u' giao hàng ', u' síp ': u' giao hàng ',
        ' beautiful ': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',
        u' order ': u' đặt hàng ',
        ' chất lg ': u' chất lượng ', u' sd ': u' sử dụng ', u' dt ': u' điện thoại ', u' nt ': u' nhắn tin ',
        u' sài ': u' xài ', u' bjo ': u' bao giờ ',
        ' thik ': u' thích ', u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',
        u' quả ng ': u' quảng ',
        ' dep ': u' đẹp ', u' xau ': u' xấu ', ' delicious ': u' ngon ', u' hàg ': u' hàng ', u' qủa ': u' quả ',
        ' iu ': u' yêu ', ' fake ': u' giả mạo ', ' trl ': ' trả lời ', ' >< ': u' positive ',
        ' por ': u' tệ ', ' poor ': u' tệ ', ' ib ': u' nhắn tin ', ' rep ': u' trả lời ', u' fback ': ' feedback ',
        ' fedback ': ' feedback '}

    for k, v in replace_list.items():
        text = text.replace(k, v)
    # xóa số  ra khỏi chuỗi
    text = re.sub(r"\d+", "", text)

    return text


def normalize_text(text):
    """Chuẩn hóa chuỗi"""
    # Chuyển thành chữ thường
    text = text.lower().replace('\n', ' ')

    text = emoj_to_string(text)
    text = normalizeString(text)
    text = emoj_to_string(text)

    text = remove_punctuation(text)
    text = replace_abbreviations(text)

    # tách từ
    text = ViTokenizer.tokenize(text)
    texts = text.split()
    len_text = len(texts)
    for i in range(len_text):
        # Xử lý vấn đề phủ định
        # "Món này chẳng ngon chút nào , k quá tệ --> món này notpositive chút nào notnegative
        if texts[i] in not_list:
            for j in range(i,len_text-1):
                if texts[j + 1] in pos_list:
                    texts[i] = 'notpositive'
                    texts[j + 1] = ''
                    break

                if texts[j + 1] in nag_list:
                    texts[i] = 'notnagative'
                    texts[j + 1] = ''
                    break

        else:  # Thêm feature cho những từ positive (món này ngon--> món này ngon positive)
            if texts[i] in pos_list:
                texts.append('positive')
            elif texts[i] in nag_list:
                texts.append('nagative')

    text = ' '.join([text for text in texts if text != ''])

    return text


def create_stopwords():
    stopwords = []
    with open('stopwords.txt', 'r') as f:
        for line in f.readlines():
            word = line.strip()
            stopwords.append('_'.join(word.split(' ')))
    f.close()
    return stopwords


def load_data_format(filename1):
    with open(filename1, 'r') as file:
        contents = json.load(file)
        file.close()
    return contents


def transform_to_dataset(x_set, y_set):
    X, y = [], []
    for document, topic in zip(list(x_set), list(y_set)):
        document = normalize_text(document)
        X.append(document.strip())
        y.append(topic)
    return X, y


def prepare_data():
    train_data = pd.DataFrame(load_data_format('data/data.json'))

    X_train, X_test, y_train, y_test = train_test_split(train_data.comment, train_data.label, test_size=0.3,
                                                        random_state=42)
    X_train, y_train = transform_to_dataset(X_train, y_train)
    X_test, y_test = transform_to_dataset(X_test, y_test)

    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train, stopwords):
    tfidfVectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords, max_features=30000, max_df=0.5, min_df=5,
                                      ngram_range=(1, 3), norm='l2', smooth_idf=True)

    classifier_1 = LinearSVC(fit_intercept=True, C=1.4)
    classifier_2 = MultinomialNB(alpha=1.0)

    steps = [('tfidfVectorizer', tfidfVectorizer),
             ('classifier', classifier_1)]

    clf = Pipeline(steps)
    clf.fit(X_train, y_train)
    clf = joblib.dump('model.pkl')
    return clf



def test(file_path, model):
    test_data = pd.DataFrame(load_data_format(file_path))
    X_test, y_test = transform_to_dataset(test_data.comment, test_data.label)
    y_pred = model.predict(X_test)
    report1 = metrics.classification_report(y_test, y_pred, labels=[1, 0])
    classifier = model['classifier']
    print('Name of classifier : ', type(classifier).__name__)
    print(report1)


# X_train, y_train, X_test, y_test = prepare_data()
# stopwords = create_stopwords()

clf = joblib.load('model.pkl')
test('data/test_data.json', clf)
#
# while True:
#   print('Hãy nhập vào 1 comment : ')
#   X_test = input()
#   X_test = normalize_text(X_test.strip())
#   print([X_test])
#
#   y_predict = clf.predict([X_test])
#   print('Comment được phân loại là : ')
#   if int(y_predict[0]) == 0:
#     print('Tích cực')
#   else:
#     print('Tiêu cực')
#   print()


# def load_model():
#     import joblib
#     model = joblib.load('model.pkl')
#     return model
#
# def classify_one_comment(model,comment):
#     comment = normalize_text(comment)
#     print(comment)
#     predict = model.predict([comment])
#     return predict

