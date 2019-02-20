import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from gensim.models import KeyedVectors
from keras.initializers import Constant
import tensorflowjs as tfjs


vocabulary_size = 50000
max_words = 500

if __name__ == "__main__":
    word_embeddings = KeyedVectors.load_word2vec_format('model/pq-150d-50K.vec')

    # load the dataset
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    word_index = imdb.get_word_index()
    embedding_dims = word_embeddings.vector_size
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))

    for word, i in imdb.get_word_index().items():
        try:
            embedding_vector = word_embeddings.get_vector(word)
            embedding_matrix[i] = embedding_vector
        except KeyError:
            continue


    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model_with_embeddings = Sequential()
    model_with_embeddings.add(Embedding(len(word_index) + 1, embedding_dims, input_length=max_words, trainable=False, embeddings_initializer=Constant(embedding_matrix)))
    model_with_embeddings.add(model)
    model_with_embeddings.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    batch_size = 64
    num_epochs = 3

    X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
    X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

    model_with_embeddings.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)
    tfjs.converters.save_keras_model(model, 'model/lstm/')

    scores = model_with_embeddings.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])



