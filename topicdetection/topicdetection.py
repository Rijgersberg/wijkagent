import re
from time import time
from datetime import datetime
from collections import OrderedDict, defaultdict
import pickle

import numpy as np
import pandas as pd
import gensim
from gensim import corpora
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer

MODELID = 0


def main(group):
    load_data_and_calculate_topics(group)


def load_data_and_calculate_topics(group, date_from=None, date_to=None):
    num_topics = 10
    num_top_words = 3
    group_df, conversations, chatlog = load_data(group, date_from, date_to)

    global MODELID
    MODELID += 1

    topics = get_topics(group_df, conversations, num_topics, num_top_words)
    result = {'group': group,'modelid': MODELID,
              'topics': topics}
    filename = 'stored_results/{}.pickle'.format(MODELID)
    with open(filename, 'wb') as flo:
        pickle.dump(result, flo)

    filename = 'stored_results/data_{}.pickle'.format(MODELID)
    with open(filename, 'wb') as flo:
        pickle.dump({'group_df':group_df}, flo)
    return result


def get_docs_for_topic(modelid, topic):
    filename = 'stored_results/{}.pickle'.format(modelid)
    with open(filename, 'rb') as flo:
        result = pickle.load(flo)

    filename = 'stored_results/data_{}.pickle'.format(modelid)
    with open(filename, 'rb') as flo:
        data = pickle.load(flo)
        group_df = data['group_df']
        topic2doc = data['topic2doc']
    
    print(topic2doc)
    docs = topic2doc[topic]
    cols_to_keep = ['datetimes', 'names', 'contents', 'conversation']
    df_filtered = group_df.loc[group_df['conversation'].isin(docs), cols_to_keep]
    return df_filtered.to_dict(orient='records')


def get_message_count(group_df, docs):
    return len(group_df.loc[group_df['conversation'].isin(docs)])


def load_data(group, date_from=None, date_to=None):
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

    with open(chatlog['path'], encoding="utf8") as chatlog_file:
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

    # date filter
    if date_from is not None:
        df = df.loc[df['datetimes'] >= date_from, :]
    if date_to is not None:
        df = df.loc[df['datetimes'] <= date_to, :]

    conversations = []
    for conversation_number in df['conversation'].unique():
        conversations.append(" ".join(df.loc[df['conversation'] == conversation_number, 'contents'].tolist()))

    return (df, conversations, chatlog)


def get_topics(group_df, conversations, num_topics, num_top_words):
    print("Extracting tf-idf features for NMF...")
    n_features = 10000
    stopwoorden = get_stopwoorden()
    vectorizer = TfidfVectorizer(binary=False, max_df=0.7, ngram_range=(4, 4),
                                 max_features=n_features, stop_words=stopwoorden)
    t0 = time()
    tfidf = vectorizer.fit_transform(conversations)
    vocab = np.array(vectorizer.get_feature_names())
    print("done in %0.3fs." % (time() - t0))

    clf = decomposition.NMF(n_components=num_topics, random_state=1)
    doctopic = clf.fit_transform(tfidf)

    # doctopic = doctopic / np.sum(doctopic, axis=1, keepdims=True)
    print(doctopic)

    doc2topic = list(np.argmax(doctopic, axis=1))
    topic2doc = defaultdict(list)
    for doc, topic in enumerate(doc2topic):
        topic2doc[topic].append(doc)
    topic2doc = dict(topic2doc)

    topics = []
    for i, topic in enumerate(clf.components_):
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topics.append({'terms': ['' + vocab[i] for i in word_idx],
                            'conversationNrs': topic2doc[i],
                            'messageCount': get_message_count(group_df, topic2doc[i])})

    print("")
    print("")
    print("========== Automatisch gevonden onderwerpen in de chat ==========")
    print("")
    for t, topic in enumerate(topics, start=1):
        print("Onderwerp {}: {}".format(t, ' | '.join(topic['terms']))) 
    return topics


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
