import re
from time import time

import numpy as np
import pandas as pd
import gensim
from gensim import corpora
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    chatlog_sets = {'floor': {'path': 'data/WhatsApp-chat floor_fixed.txt',
                          'date_format': 'nl-short'},
                'werk': {'path': 'data/WhatsApp Chat - werk.txt',
                          'date_format': 'us'},
                'noodgroep': {'path': 'data/WhatsApp Chat - noodgroep.txt',
                          'date_format': 'nl'},
                          }
    chatlog = chatlog_sets['werk']

    with open(chatlog['path']) as chatlog_file:
        chatlog_string = chatlog_file.read()

    chatlog_string = strip_newlines(chatlog_string, date_format=chatlog['date_format'])

    contents = []
    for line in chatlog_string.splitlines():
        try:
            datetime, name, content = line.strip().split(': ', 2)
            contents.append(content)
        except ValueError as e:
            print("Parse error:" + line)

    print(contents[:5])

    # stopwoorden = get_stopwoorden()
    # texts = [[word for word in content.lower().split() if word not in stopwoorden]
    #          for content in contents]

    # dictionary = corpora.Dictionary(texts)
    # corpus = [dictionary.doc2bow(text) for text in texts]
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
    # print(ldamodel.print_topics(num_topics=5, num_words=5))

    print("Extracting tf-idf features for NMF...")
    n_features = 10000
    stopwoorden = get_stopwoorden()
    vectorizer = TfidfVectorizer(max_df=0.1, ngram_range=(1, 2), max_features=n_features, stop_words=stopwoorden)
    t0 = time()
    tfidf = vectorizer.fit_transform(contents)
    vocab = np.array(vectorizer.get_feature_names())
    print("done in %0.3fs." % (time() - t0))

    num_topics = 10
    num_top_words = 20
    clf = decomposition.NMF(n_components=num_topics, random_state=1)

    doctopic = clf.fit_transform(tfidf)

    topic_words = []
    for topic in clf.components_:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append(['' + vocab[i] for i in word_idx])

    print("")
    for t in range(len(topic_words)):
        print("Topic {}: {}".format(t + 1, ' | '.join(topic_words[t][:5])))

    # # Use tf (raw term count) features for LDA.
    # print("Extracting tf features for LDA...")
    # tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
    #                                 max_features=n_features,
    #                                 stop_words='english')
    # t0 = time()
    # tf = tf_vectorizer.fit_transform(data_samples)
    # print("done in %0.3fs." % (time() - t0))

    # Fit the NMF model
    # print("Fitting the NMF model with tf-idf features, "
    #       "n_samples=%d and n_features=%d..."
    #       % (n_samples, n_features))
    # t0 = time()
    # nmf = NMF(n_components=n_topics, random_state=1,
    #           alpha=.1, l1_ratio=.5).fit(tfidf)
    # print("done in %0.3fs." % (time() - t0))

    # print("\nTopics in NMF model:")
    # tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # print_top_words(nmf, tfidf_feature_names, n_top_words)


def strip_newlines(string, date_format='nl'):
    if date_format == 'nl':
        pattern = r'\n(?:(?!(\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})))'
    elif date_format == 'nl-short':
        pattern = r'\n(?:(?!(\d{2}-\d{2}-\d{2}, \d{2}:\d{2})))'
    elif date_format == 'us':
        pattern = r'\n(?:(?!(\d{2}\/\d{2}\/\d{4}, \d{2}:\d{2}:\d{2})))'
    else:
        raise Exception('Unknown date_format')
    
    prog = re.compile(pattern)
    return prog.sub(' ', string)

def get_stopwoorden():
    file = 'data/stopwoorden.txt'
    with open(file) as flo:
        return flo.read().splitlines() 


def parse_dates(x):
    return x


if __name__ == '__main__':
    main()


