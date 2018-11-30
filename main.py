import logging
import json
import codecs
import os

from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pqkmeans


def reduce_dimensions_pca(vectors, dimensions=150):
    reduced_vectors = PCA(n_components=dimensions).fit_transform(vectors)
    return reduced_vectors


def product_quantize(vectors, subdims=30, centres=1000):
    encoder = pqkmeans.encoder.PQEncoder(num_subdim=subdims, Ks=centres)
    encoder.fit(vectors)
    vectors_pq = encoder.transform(vectors)
    reconstructed_vectors = encoder.inverse_transform(vectors_pq)
    return reconstructed_vectors, vectors_pq, encoder.codewords


def save_matrix(name, matrix):
    matrix_shape = list(matrix.shape)
    matrix_list = matrix.flatten().tolist()
    data = {
        "shape": matrix_shape,
        "vectors": matrix_list
    }
    with open(os.path.join('model', "{:s}.json".format(name)), 'w') as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = KeyedVectors.load_word2vec_format('data/wiki-news-300d-50k.vec')
    embeddings = model.vectors
    size = embeddings.nbytes

    print("Reduce dimensions using PCA...")
    embeddings = reduce_dimensions_pca(embeddings)
    print("Compress embeddings using product quantization...")
    embeddings, embeddings_pq, codewords = product_quantize(embeddings)

    print("Calculating accuracy...")
    words = [model.index2word[idx] for idx in range(len(embeddings))]
    model = KeyedVectors(vector_size=embeddings.shape[1])
    model.add(words, embeddings, replace=True)
    model.evaluate_word_analogies('questions-words.txt', restrict_vocab=50000)

    new_size = embeddings_pq.nbytes + codewords.nbytes
    print("Size reduction: {:f}%".format((size-new_size)*100/size))

    if not os.path.exists("model"):
        os.makedirs("model")

    print("Saving vocabulary...")
    with codecs.open('model/vocab.json', 'w', 'UTF-8') as f:
        f.write(json.dumps(words))

    print("Saving codewords...")
    save_matrix('codewords', codewords)

    print("Saving embeddings...")
    save_matrix('embeddings', embeddings_pq)


