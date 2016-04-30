# -*- coding: UTF-8 -*-
from __future__ import division
import gensim
import re
import pymorphy2
import nltk.data
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from nltk.stem.snowball import SnowballStemmer

model = gensim.models.Word2Vec.load('s100features_4minwords_10context_bigrams_1e-3')
model_words = set(model.index2word)
stemmer = SnowballStemmer("russian")
bigram_transformer = gensim.models.Phrases.load('bigram_transformer_1')
obscene_list = [u'иди_нахуй', u'сдохни', u'заткнись', u'мудак']

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#morph = pymorphy2.MorphAnalyzer()
f = open('specom-positive')
r_pos = list(set(f.readlines()))
f.close()

f = open('specom-negative')
r_neg = list(set(f.readlines()))
f.close()

def avg(l):
    if l:
        return sum([len(s) for s in l])/len(l)
    else:
        return 0


def sentence_semantic_difference(sentence):
    in_model = [word for word in sentence if word in model_words]
    scores = []
    if len(in_model) < 2:
        return [0]
    for i in range(len(in_model)-2):
        for k in range(1,len(in_model)-1):
            scores.append(model.similarity(in_model[i],in_model[k]))
    if scores:
        return [sum(scores)/len(scores)]
    else:
        return [0]


def message_to_wordlist(message, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    # review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove messages numbers
    tokenizer = WordPunctTokenizer()
    # 3. Convert words to lower case and split them
    words = tokenizer.tokenize(message)
    lemmas = []
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("russian"))
        words = [w.lower() for w in words if w not in stops and len(w)>2]
        words = [re.sub(">>\d+|\.|,", "", w) for w in words]
        words = [re.sub(u"ё", 'e', w, re.UNICODE) for w in words]
        words = [stemmer.stem(w) for w in words]
        words = [w for w in words if w.isalpha()]
    #for word in words:
    #    word_parsed = morph.parse(word)
    #    if len(word_parsed) > 0:
    #        lemmas.append(word_parsed[0].normal_form)
    # 5. Return a list of words
    lemmas = bigram_transformer[words]
    return lemmas

# Define a function to split a message into parsed sentences
def message_to_sentences( message, tokenizer, remove_stopwords=True ):
    sentences = []
    # Function to split a message into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    if type(message) == str:
        message = message.decode('utf-8')
        raw_sentences = tokenizer.tokenize(message.strip())
        #
        # 2. Loop over each sentence
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call message_to_wordlist to get a list of words
                sentences.append( message_to_wordlist( raw_sentence, \
              remove_stopwords ))

    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    sentences = [s for s in sentences if s]
    return sentences

def message_find_sentences(mess):
    sentences = []
    for line in mess:
        message = re.findall('(<.*?>)(.*?)(<.*)',line)
        if message:
            message = message[0][1]
            sentences.append(message_to_sentences(message, tokenizer, True))
    return sentences

sentences_pos = [message_find_sentences(s) for s in r_pos]
sentences_neg = [message_find_sentences(s) for s in r_neg]

error_words = []
pos = 0
neg = 0
err = 0
fail = 0
sent_pos_vals = []
for message in sentences_pos:
    scores = []
    if message:
        for word in message:
            if word in model_words:
                score = model.n_similarity([word], obscene_list)
                scores.append(score)
            else:
                error_words.append(word)
        word = morph.parse(word)[0].normal_form
        try:
            score = model.n_similarity([word], obscene_list)
            scores.append(score)
        except:
            scores.append(0)
        res = [max(scores)-min(scores)] + [max(scores)] + [min(scores)] + [sum(scores)/len(scores)] + [len(message)] + [max([len(w) for w in message])] + [min([len(w) for w in message])] + [avg(message)] + sentence_semantic_difference(message)
        sent_pos_vals.append(res)
        if len(scores) != 0:
            score = sum(scores)/len(scores)
            if score<=0.4:
                pos+=1
            elif score == 0:
                fail += 1
            else:
                err +=1
        else:
            fail+=1
    else:
        fail+=1

sent_neg_vals = []
for message in sentences_neg:
    scores = []
    if message:
        for word in message:
            if word in model_words:
                score = model.n_similarity([word], obscene_list)
                scores.append(score)
            else:
                error_words.append(word)
        word = morph.parse(word)[0].normal_form
        try:
            score = model.n_similarity([word], obscene_list)
            scores.append(score)
        except:
            scores.append(0)
        res = [max(scores)-min(scores)] + [max(scores)] + [min(scores)] + [sum(scores)/len(scores)] + [len(message)] + [max([len(w) for w in message])] + [min([len(w) for w in message])] + [avg(message)] + sentence_semantic_difference(message)
        sent_neg_vals.append(res)
        if len(scores) != 0:
            score = sum(scores)/len(scores)
            if score>=0.4:
                neg+=1
            elif score == 0:
                fail += 1
            else:
                err +=1
        else:
            fail+=1
    else:
        fail+=1
print (0.3)
print(pos)
print(neg)
print('error',err)
#print(fail)
#print('len_pos', len(sentences_pos))
#print('len_neg', len(sentences_neg))
print (100*(pos+neg)/(len(sentences_pos) + len(sentences_neg)-fail))


print('random_forest')
sent_pos_slice = int(round(len(sent_pos_vals)*0.9))
sent_neg_slice = int(round(len(sent_neg_vals)*0.9))
train = sent_pos_vals[:sent_pos_slice] + sent_neg_vals[:sent_neg_slice]
train_values = []
test = sent_pos_vals[sent_pos_slice:] + sent_neg_vals[sent_neg_slice:]
test_sents = sent_pos_vals[sent_pos_slice:] + sent_neg_vals[sent_neg_slice:]
target = [1] * sent_pos_slice + [0] * sent_neg_slice
test_res = [1]*(len(sent_pos_vals) - sent_pos_slice) + [0]*(len(sent_neg_vals) - sent_neg_slice)
rf = RandomForestClassifier(n_estimators = 100, n_jobs=2)
print('training random forest')
train = pd.DataFrame(train)
train = train.fillna(0)
rf.fit(train, target)
test = pd.DataFrame(test)
test = test.fillna(0)
prediction = rf.predict(test)
score = 0
for i in range(len(prediction)-1):
    if prediction[i] == test_res[i]:
        score +=1
print ('rf result is ', 100*score/len(prediction))
print rf.feature_importances_
