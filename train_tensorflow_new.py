# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import itertools 
import csv
import gensim
import re
import nltk.data
import tensorflow
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras import backend as K

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """	
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def message_to_wordlist(message, lemmas_bool, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove messages numbers
    message_text = re.sub(">>\d+","", message)
    message_text = message_text.lower()
    message_text = re.sub(u"ё", 'e', message_text, re.UNICODE)
    message_text = clean_str(message_text)
    tokenizer = WordPunctTokenizer()
    # 3. Convert words to lower case and split them
    words = tokenizer.tokenize(message_text)
    lemmas = []
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    if lemmas_bool == 'l':
        for word in words:
            word_parsed = morph.parse(word)
            if len(word_parsed) > 0:
                lemmas.append(word_parsed[0].normal_form)
    elif lemmas_bool == 's':
        for word in words:
            word = stemmer.stem(word)
            if len(word) > 0:
                lemmas.append(word)
    else:
        lemmas = words
    # 5. Return a list of words
    return(lemmas)
    #return(words)

# Define a function to split a message into parsed sentences
def message_to_sentences( message, tokenizer, lemmas_bool, remove_stopwords=False):
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
                sentences.append(message_to_wordlist( raw_sentence,lemmas_bool, remove_stopwords))
    return sentences


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    new_labels = []
    for label in labels:
        if label == 1:
            new_labels.append([1,0])
        else:
            new_labels.append([0,1])
    labels = new_labels
    y = np.array(labels)
    return [x, y]

def load_data():
    messages = pd.read_csv( 'aggression.csv', header=0,
 delimiter="\t", quoting = csv.QUOTE_MINIMAL ) 
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    labels = messages[:]['Aggression']
    messages = messages[:]['Text']
    messages = [message_to_sentences(message, tokenizer, '') for message in messages]
    len_messages = [len(m) for m in messages]
    max_message = max(len_messages)
    print(max_message)
    # pad messages
    messages = [message + [['<PAD/>']]*(max_message - len(message)) for message in messages]
    len_sentences = [len(s) for m in messages for s in m]
    max_sent = max(len_sentences)
    print (max_sent)
    messages = [[s + ['<PAD/>'] * (max_sent - len(s)) for s in m] for m in messages]  # turn to the same length
    vocabulary, vocabulary_inv = build_vocab([s for sents in messages for s in sents])
    x, y = build_input_data(messages, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, max_message, max_sentence, messages]

np.random.seed(2)

model_variation = 'CNN-non-static'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

# Model Hyperparameters
sequence_length = 18
embedding_dim = 600          
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# Training parameters
batch_size = 32
num_epochs = 100
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 4  # Minimum word count                        
context = 10        # Context window size    

print("Loading data...")
x, y, vocabulary, vocabulary_inv = load_data()

if model_variation == 'CNN-non-static' or model_variation == 'CNN-static':
    embedding_model = gensim.models.Word2Vec.load('model')
    model_words = embedding_model.index2word
    embedding_weights = [np.array([embedding_model[w] if w in vocabulary and w in model_words\
                                                        else np.random.uniform(-0.25,0.25,600)\
                                                        for w in vocabulary_inv])]
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')   

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices].argmax(axis=1)

print("Vocabulary Size: {:d}".format(len(vocabulary)))

# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel
graph = Graph()
graph.add_input(name='input', input_shape=(sequence_length, embedding_dim))
graph.add_input(name='input', input_shape=(sequence_length, pos_length))
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)
    pool = MaxPooling1D(pool_length=2)
    graph.add_node(conv, name='conv-%s' % fsz, input='input')
    graph.add_node(pool, name='maxpool-%s' % fsz, input='conv-%s' % fsz)
    graph.add_node(Flatten(), name='flatten-%s' % fsz, input='maxpool-%s' % fsz)

if len(filter_sizes)>1:
    graph.add_output(name='output',
                     inputs=['flatten-%s' % fsz for fsz in filter_sizes],
                     merge_mode='concat')
else:                 
    graph.add_output(name='output', input='flatten-%s' % filter_sizes[0])

# main sequential model
model = Sequential()
if not model_variation=='CNN-static':
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode='binary')

# Training model
# ==================================================
model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
          nb_epoch=num_epochs, show_accuracy=True,
          validation_split=val_split, verbose=2)