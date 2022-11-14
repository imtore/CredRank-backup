import os
import os.path as op
import json
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.stem import PorterStemmer
import preprocess_twitter
import glove
import numpy as np

my_punctuation = string.punctuation.replace("<=>", "=")


def convert_lowercase(s):
    return s.lower()


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', my_punctuation))


def remove_whitespace(s):
    return s.strip()


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [i for i in tokens if not i in stop_words]


def apply_stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(i) for i in tokens]


def normalize(post):
    post = preprocess_twitter.tokenize(post)
    post = remove_punctuation(post)
    post = remove_whitespace(post)
    post = word_tokenize(post)
    mwtokenizer = MWETokenizer(separator='', mwes=[('<', 'hashtag', '>'), ('<', 'elong', '>'), ('<', 'user', '>'),
                                                   ('<', 'number', '>'), ('<', 'url', '>'),
                                                   ('<', 'heart', '>'), ('<', 'repeat', '>'), ('<', 'smile', '>'),
                                                   ('<', 'lolface', '>'), ('<', 'sadface', '>'),
                                                   ('<', 'neutralface', '>'),
                                                   ('<', 'allcaps', '>')])
    post = mwtokenizer.tokenize(post)
    post = remove_stopwords(post)

    return post


PATH = '../../../../FakeNewsNet/code/fakenewsnet_dataset/user_timeline_tweets'
PAGE_SIZE = 1000
user_order = pd.read_csv('../../user_order.csv', dtype={'0': str}, index_col=0)
users = list(user_order['0'])
print(users)

# TODO: time_encoding


def reduce_date(date):
    part1 = date[4:-19]
    part1 = part1[0:-1]
    return part1[0:3] + part1[4:] + date[-4:]


def collect():
    dict_of_features = {}
    num = 1
    for user in users:
        print(num)
        if num // PAGE_SIZE < 257:
            num += 1
            continue
        print('STARTING')
        with open(PATH + '/' + user + '.json') as timeline:
            posts = json.load(timeline)

            temp = {'timestamps': [], 'texts': []}
            for post in posts:

                vectors = []
                for word in normalize(post['text']):
                    if word in glove.model.vocab:
                        vectors.append(glove.model.get_vector(word))
                    else:
                        vectors.append(np.zeros(50, dtype='float32'))

                vector = np.mean(vectors, axis=0)

                # ignore nan values
                if vector.dtype == 'float64':
                    continue

                vector = np.round(vector, 5)
                temp['texts'].append(list(vector))
                temp['timestamps'].append(reduce_date(post['created_at']))

            dict_of_features[user] = temp

        num += 1
        print(num)
        if num % PAGE_SIZE == 0:
            print('PAGE: ', num // PAGE_SIZE)
            posts_df = pd.DataFrame.from_dict(dict_of_features)
            posts_df.to_csv('timeline_tweets' + str(num // PAGE_SIZE) + '.csv')
            dict_of_features = {}

    posts_df = pd.DataFrame(dict_of_features)
    posts_df.to_csv('timeline_tweets' + str(num // PAGE_SIZE + 1) + '.csv', index=False)

    return dict






if __name__ == '__main__':
    post_dict = collect()
