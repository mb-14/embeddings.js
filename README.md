# embeddings.js

Word embeddings for the the web

Word embeddings often require a large number of parameters which results in a large memory and storage footprint.
This makes deploying pre-trained word embeddings like [fastText](https://fasttext.cc/) and [GloVe](https://nlp.stanford.edu/projects/glove/) in mobile and browser environments very difficult. In this project,
we will compress pre-trained word vectors using simple post-processing techniques like **PCA dimensionality reduction** and **production quantization**.
The resulting embeddings are significantly smaller compared to the original embeddings with no considerable drop in accuracy. The final vectors along with the helper methods to access them are bundled into a javascript library.

## Demo
You can check out the demo of the js library on this page: https://mb-14.github.io/embeddings.js/demo

## Models

- [models/compressor](): Module to compress pretrained word embeddings using PCA and product quantization
- [models/sentiment_analysis]() LSTM model for sentiment classifcation trained on the sentiment140 dataset 

## Instructions

### Run on local

This project uses [yarn](https://yarnpkg.com) for dependencies 
Install dependencies and run demo
```bash
yarn
yarn run demo
```
You can then check all the demos at [http://localhost:8080]()

### Build library
Build the production version of `embeddings.js` in the `dist` folder

```bash
yarn build
```

### Generate word embeddings
Bundle the vector and vocabulary files generated using the [compressor](/models/compressor) into a single
JSON file.
Make sure the following files are present in the `--input` directory:
- centroids.json
- codes.json
- vocab.json

```bash
yarn build-embeddings --input models/compressor/generated --output output_dir/word-embeddings.json
```