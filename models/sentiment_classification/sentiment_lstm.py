import argparse
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.models import KeyedVectors
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.initializers import Constant
from keras.utils import plot_model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, AveragePooling1D, MaxPooling1D, Flatten, Dropout, GlobalMaxPooling1D, \
    LSTM
from sklearn.model_selection import train_test_split
import re
from sklearn.utils import shuffle
import tensorflowjs as tfjs


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 50000
VALIDATION_SPLIT = 0.2
EPOCHS = 6
BATCH_SIZE = 512

def load_data_set():
    data = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding='latin-1',
                       header=None)
    data.columns = ['sentiment', 'id', 'date', 'q', 'user', 'text']
    data.text = data.text.apply(lambda x:
                                ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", x).split()))
    data = data.drop(['id', 'date', 'q', 'user'], axis=1)
    data = shuffle(data)
    data = data[data.sentiment != 2]
    data.sentiment = data.sentiment / 4
    data = data[:100000]
    data = data[['text', 'sentiment']]
    data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
    for idx, row in data.iterrows():
        row[0] = row[0].replace('rt', ' ')
    # data.to_csv('data/training.100000.processed.csv', sep='\t', encoding="latin-1", index=False)
    # data = pd.read_csv("data/training.100000.processed.csv", sep='\t', encoding='latin-1')
    # data = data[['text', 'sentiment']]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tweets = data['text'].values
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y = data['sentiment'].values
    print(y[0])
    return X, y, word_index


def create_embedding_layer(word_index, embeddings_path):
    embeddings = KeyedVectors.load_word2vec_format(embeddings_path)
    embedding_dims = embeddings.vector_size
    vocabulary_size = len(embeddings.vocab)
    nb_words = min(vocabulary_size, len(word_index))

    embedding_matrix = np.zeros((nb_words, embedding_dims))

    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = embeddings.get_vector(word)
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue

    embedding_layer = Embedding(nb_words, embedding_dims, embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer, embedding_dims


def create_model(word_index, embeddings):
    embedded_sequences, embeddings_dim = create_embedding_layer(word_index, embeddings)

    model = Sequential([
        embedded_sequences,
        Dropout(0.4),
        LSTM(128),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model, embeddings_dim


def get_model_without_embeddings(model, input_shape):
    input = Input(shape=input_shape)
    model.summary()
    x = Dropout(0.4)(input)
    x = LSTM(128, weights=model.get_layer('lstm_1').get_weights())(x)
    x = Dense(64, activation='relu', weights=model.get_layer('dense_1').get_weights())(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation='sigmoid', weights=model.get_layer('dense_2').get_weights())(x)
    new_model = Model(input, y)
    new_model.summary()
    return new_model


def save_model(path, model):
    print("Saving models in the {} directory".format(path))
    if not os.path.exists(path):
        os.makedirs(path)

    tfjs.converters.save_keras_model(model, path)
    model.save(os.path.join(path, 'sentiment_lstm.h5'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="embeddings", required=True,
                        help="path to word embeddings vector file")

    args = parser.parse_args()

    X, y, word_index = load_data_set()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)
    X_valid, y_valid = X_train[:BATCH_SIZE], y_train[:BATCH_SIZE]
    X_train2, y_train2 = X_train[BATCH_SIZE:], y_train[BATCH_SIZE:]
    model, embedding_dims = create_model(word_index, args.embeddings)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[reduce_lr])
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])
    new_model = get_model_without_embeddings(model, (MAX_SEQUENCE_LENGTH, embedding_dims))
    save_model('generated', new_model)
