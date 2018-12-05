# embeddings.js

Word embeddings for the the web

Word embeddings often require a large number of parameters which results in a large memory and storage footprint.
This makes deploying pre-trained word embeddings like [fastText](https://fasttext.cc/) and [GloVe](https://nlp.stanford.edu/projects/glove/) in mobile and browser environments very difficult. In this project,
we will compress pre-trained word vectors using simple post-processing techniques like **PCA dimensionality reduction** and **production quantization**.
The resulting embeddings are significantly smaller compared to the original embeddings with no considerable drop in accuracy.

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
Saving model/vocab.json
Saving model/codewords.json
Saving model/embeddings.json
```

### Building JS library
Run `gulp build` to generate the js library and the final model JSON file in the `dist` directory



## Example
