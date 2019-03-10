#!/bin/bash

pretained_vectors_path=data/crawl-300d-50K.vec

if [[ -e "${pretained_vectors_path}" ]]; then
    echo "Vector file already exist"
    exit 1
fi

if [[ ! -e "${pretained_vectors_path}.zip" ]]; then
    mkdir data
    echo "Downloading fastText word vectors for 50k most common words..."
    curl -L https://mb-14.github.io/static/crawl-300d-50K.vec.zip > ${pretained_vectors_path}.zip
fi

unzip ${pretained_vectors_path}.zip -d data/
