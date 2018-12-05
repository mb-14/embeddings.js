import argparse
import json
import codecs
import logging
import os
import warnings

from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
import pqkmeans

warnings.simplefilter(action='ignore', category=FutureWarning)


def reduce_dimensions_pca(vectors, dimensions=150):
    reduced_vectors = PCA(n_components=dimensions).fit_transform(vectors)
    return reduced_vectors


def product_quantize(vectors, subdims=30, centres=1000):
    encoder = pqkmeans.encoder.PQEncoder(iteration=40, num_subdim=subdims, Ks=centres)
    encoder.fit(vectors)
    vectors_pq = encoder.transform(vectors)
    reconstructed_vectors = encoder.inverse_transform(vectors_pq)
    return reconstructed_vectors, vectors_pq, encoder.codewords


def compute_accuracy(model):
    print("Calculating accuracy...")
    accuracy, _ = model.evaluate_word_analogies('questions-words.txt', restrict_vocab=50000)
    print("Accuracy: {:f}%".format(accuracy*100))

def save_matrix(name, matrix):
    matrix_shape = list(matrix.shape)
    matrix_list = matrix.flatten().tolist()
    data = {
        "shape": matrix_shape,
        "vectors": matrix_list
    }
    file = os.path.join("model", "{:s}.json".format(name))
    with open(file, 'w') as f:
        print("Saving {:s}".format(file))
        f.write(json.dumps(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="filename", required=True,
                        help="input file containing word embeddings")
    parser.add_argument("-c", dest="compress", help="compress word embeddings", action="store_true")
    parser.add_argument("-v", dest="verbose", help="verbosity", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print("Loading vectors in gensim...")
    model = KeyedVectors.load_word2vec_format(args.filename)

    if not args.compress:
        compute_accuracy(model)
        exit(0)

    embeddings = model.vectors
    size = embeddings.nbytes

    print("Reduce dimensions using PCA...")
    embeddings = reduce_dimensions_pca(embeddings)
    new_size = embeddings.nbytes
    print("Size reduction: {:f}%".format((size - new_size) * 100 / size))

    print("Compress embeddings using product quantization...")
    embeddings, embeddings_pq, codewords = product_quantize(embeddings)

    new_size = embeddings_pq.nbytes + codewords.nbytes
    print("Size reduction: {:f}%".format((size-new_size)*100/size))

    words = [model.index2word[idx] for idx in range(len(embeddings))]
    model = KeyedVectors(vector_size=embeddings.shape[1])
    model.add(words, embeddings, replace=True)
    compute_accuracy(model)

    if not os.path.exists("model"):
        os.makedirs("model")

    vocab_file_path = os.path.join("model", "vocab.json")
    with codecs.open(vocab_file_path, 'w', 'UTF-8') as f:
        print("Saving {:s}".format(vocab_file_path))
        f.write(json.dumps(words))

    save_matrix('codewords', codewords)
    save_matrix('embeddings', embeddings_pq)


