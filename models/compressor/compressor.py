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


def save_matrix(file, matrix):
    matrix_shape = list(matrix.shape)
    matrix_list = matrix.flatten().tolist()
    data = {
        "shape": matrix_shape,
        "vectors": matrix_list
    }
    with open(file, 'w') as f:
        print("Saving {:s}".format(file))
        f.write(json.dumps(data))


def human_format(num):
    if not os.path.exists("generated"):
        os.makedirs("generated")

    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def save_model(path, embedding_size, word_list, codes, centroids):
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = "embeddings-{}d-{}.vec".format(embedding_size, human_format(len(word_list)))
    model.save_word2vec_format(os.path.join(path, model_name))

    vocab_file_path = os.path.join(path, "vocab.json")
    with codecs.open(vocab_file_path, 'w', 'UTF-8') as f:
        print("Saving {:s}".format(vocab_file_path))
        f.write(json.dumps(word_list))

    save_matrix(os.path.join(path, 'codes.json'), codes)
    save_matrix(os.path.join(path, 'centroids.json'), centroids)


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
    embeddings, codes, centroids = product_quantize(embeddings)

    new_size = codes.nbytes + centroids.nbytes
    print("Size reduction: {:f}%".format((size-new_size)*100/size))

    words = [model.index2word[idx] for idx in range(len(embeddings))]
    model = KeyedVectors(vector_size=embeddings.shape[1])
    model.add(words, embeddings, replace=True)
    compute_accuracy(model)

    save_model('generated', embeddings.shape[1], words, codes, centroids)


