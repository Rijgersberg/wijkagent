import re
from time import time
from datetime import datetime

import numpy as np
import pandas as pd
import gensim
from gensim import corpora
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer


def main(group):
    load_data_and_calculate_topics(group)


def load_data_and_calculate_topics(group):
    num_topics = 10
    num_top_words = 5
    group_df, conversations, chatlog = load_data(group)
    return get_topics(conversations, num_topics, num_top_words)


def load_data(group):
    chatlog_sets = {'floor': {'path': 'data/WhatsApp-chat floor_fixed.txt',
                          'date_pattern': 'nl-short',
                          'date_format': '%d-%m-%y, %H:%M'},
                'werk': {'path': 'data/WhatsApp Chat - werk.txt',
                          'date_pattern': 'us',
                          'date_format': '%d/%m/%Y, %H:%M:%S'},
                'noodgroep': {'path': 'data/WhatsApp Chat - noodgroep.txt',
                          'date_pattern': 'nl',
                          'date_format': '%d-%m-%y %H:%M:%S'},
                'eilandenbuurt': {'path': 'data/WhatsApp-chat met Eilandenbuurt Noord Oost.txt',
                          'date_pattern': 'nl-short',
                          'date_format': '%d-%m-%y, %H:%M'},
                'eilandenbuurt_opgeheven': {'path': 'data/WhatsApp-chat met Opgeheven... Noord-Oost.txt',
                          'date_pattern': 'nl-short',
                          'date_format': '%d-%m-%y, %H:%M'},
                }
    chatlog = chatlog_sets[group]

    with open(chatlog['path']) as chatlog_file:
        chatlog_string = chatlog_file.read()

    chatlog_string = strip_newlines(chatlog_string, date_pattern=chatlog['date_pattern'])

    contents = []
    datetimes = []
    names = []
    for line in chatlog_string.splitlines():
        try:
            datetime, name, content = line.strip().split(': ', 2)
            datetimes.append(parse_date(datetime, chatlog['date_format']))
            names.append(name)
            contents.append(content)
        except ValueError as e:
            print("Parse error:" + line)

    print(contents[:5])
    df = pd.DataFrame(data={'datetimes': datetimes, 'names': names, 'contents': contents})
    df['timediff'] = (df['datetimes'] - df['datetimes'].shift()).fillna(0)
    df['conversation'] = (df['timediff'] > pd.Timedelta('4 hours')).cumsum() + 1

    conversations = []
    for conversation_number in df['conversation'].unique():
        conversations.append(" ".join(df.loc[df['conversation'] == conversation_number, 'contents'].tolist()))

    return (df, conversations, chatlog)


def get_topics(conversations, num_topics, num_top_words):
    print("Extracting tf-idf features for NMF...")
    n_features = 10000
    stopwoorden = get_stopwoorden()
    vectorizer = TfidfVectorizer(binary=False, max_df=0.7, ngram_range=(1, 3),
                                 max_features=n_features, stop_words=stopwoorden)
    t0 = time()
    tfidf = vectorizer.fit_transform(conversations)
    vocab = np.array(vectorizer.get_feature_names())
    print("done in %0.3fs." % (time() - t0))

    clf = decomposition.NMF(n_components=num_topics, random_state=1)
    doctopic = clf.fit_transform(tfidf)

    topic_words = []
    for topic in clf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append(['' + vocab[i] for i in word_idx])

    print("")
    print("")
    print("========== Automatisch gevonden onderwerpen in de chat ==========")
    print("")
    for t in range(len(topic_words)):
        print("Onderwerp {}: {}".format(t + 1, ' | '.join(topic_words[t])))
    return topic_words


def strip_newlines(string, date_pattern='nl'):
    if date_pattern == 'nl':
        pattern = r'\n(?:(?!(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})))'
    elif date_pattern == 'nl-short':
        pattern = r'\n(?:(?!(\d{2}-\d{2}-\d{2}, \d{2}:\d{2})))'
    elif date_pattern == 'us':
        pattern = r'\n(?:(?!(\d{2}\/\d{2}\/\d{4}, \d{2}:\d{2}:\d{2})))'
    else:
        raise Exception('Unknown date_format')
    
    prog = re.compile(pattern)
    return prog.sub(' ', string)

def get_stopwoorden():
    file = 'data/stopwoorden.txt'
    with open(file) as flo:
        return flo.read().splitlines() 


def parse_date(datestring, date_format):
    return datetime.strptime(datestring, date_format)


if __name__ == '__main__':
    main(group='eilandenbuurt_opgeheven')
