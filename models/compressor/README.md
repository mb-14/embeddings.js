# Vector compression techniques for word embeddings

We will explore two techniques to compress our word embeddings:
- PCA dimensionality reduction
- Product quantization

## Demo
You can find the demo on this [page](https://mb-14.github.io/embeddings.js/models/compressor/demo)

## Instructions
### Downloading pre-trained word vectors
Run `fetch_vectors.sh` to download pre-trained vectors from fastText trained on the common crawl dataset. The file contains 50k word vectors, each of 300 dimensions.
```console
foo@bar:~$ ./fetch_vectors.sh
Downloading fastText word vectors for 50k most common words...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 35.5M  100 35.5M    0     0  1272k      0  0:00:28  0:00:28 --:--:-- 1278k
Archive:  data/crawl-300d-50K.vec.zip
  inflating: data/crawl-300d-50K.vec

```
### Loading vectors into gensim
Run `main.py` to load the words vectors into gensim and calculate the accuracy using the word analogies test.
```console
foo@bar:~$ python main.py -i data/crawl-300d-50K.vec
Loading vectors in gensim...
Calculating accuracy...
Accuracy: 87.078803%
``` 

### Compressing embeddings
Add the `-c` parameter to compress the embeddings and re-calculate the accuracy. The compressed embeddings, codebook of centroids and vocabulary are saved in the model directory as JSON files.

```console
foo@bar:~$ python main.py -c -i data/crawl-300d-50K.vec
Loading vectors in gensim...
Reduce dimensions using PCA...
Size reduction: 50.000000%
Compress embeddings using product quantization...
Size reduction: 93.000000%
Calculating accuracy...
Accuracy: 81.957310%
Saving generated/vocab.json
Saving generated/codes.json
Saving generated/centroids.json
```

### Generate final model file
Compress the files using lz compression, bundle everything into a single JSON file
```console
# Run this from the project root
yarn build-embeddings --input models/compressor/generated --output pretrained/word-embeddings.json
```
