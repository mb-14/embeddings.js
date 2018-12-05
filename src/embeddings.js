import * as utils from './utils';
import * as np from 'numjs';

class WordEmbeddings {

	constructor(embeddings, codewords, vocabulary) {
		var embeddingsMap = {};
		for (var i =0; i < vocabulary.length; i++) {
			embeddingsMap[vocabulary[i]] = embeddings.pick(i);
		}
		this.vocabulary = vocabulary;
		this.codewords = codewords;
		this.embeddingsMap = embeddingsMap;
	}

	// _getVector returns the vector representation of a word as an NDArray
	_getVector(word) {
		var embedding = this.embeddingsMap[word];
		var vector = this.codewords.pick(0, embedding.get(0));
		for (var i = 1; i < embedding.shape[0]; i++) {
			var codeword = this.codewords.pick(i, embedding.get(i));
			vector = np.concatenate(vector, codeword);
		}
		return vector;
	}

	// _getVector returns the vector representation of a word as a float array
	getVector(word) {
		return this._getVector(word).tolist();
	}

	// getCosineDistance returns the cosine distance between two word vectors
	getCosineDistance(word1, word2) {
		var vec1 = this._getVector(word1);
		var vec2 = this._getVector(word2);
		var dotProduct = np.dot(vec1, vec2).get(0);
		var abs1 = Math.sqrt(np.dot(vec1, vec1).get(0));
		var abs2 = Math.sqrt(np.dot(vec2, vec2).get(0));
		var cosineDistance = dotProduct/(abs1*abs2);
		return cosineDistance;
	}

	// getNearestNeighbors returns the closest k word vectors from a given word vector 
	async getNearestNeighbors(word, k=5) {
		var neighbors = [];
		var embedding = this.embeddingsMap[word];
		var vector = this._getVector(word);
		var abs = Math.sqrt(np.dot(vector, vector).get(0));

		// Precompute distances
		var lookupTable = this._computeDistances(embedding);
	
		for(var i = 0; i < this.vocabulary.length; i++) {
			var word1 = this.vocabulary[i];

			if (word1 === word) {
				continue;
			}

			var embedding1 = this.embeddingsMap[word1];
			var dotProduct = 0;
			var abs1 = 0;
			for (var j = 0; j < this.codewords.shape[0]; j++) {
				var center = embedding1.get(j);
				dotProduct += lookupTable.get(j, center, 0);
				abs1 += lookupTable.get(j, center, 1);
			}

			var neighbor = {
				distance: dotProduct/(abs*Math.sqrt(abs1)),
				word: word1,
			};

			neighbors.push(neighbor);
		}

		neighbors.sort(function(a, b) {
			if (a.distance < b.distance) {
				return 1;
			}
			if (a.distance > b.distance) {
				return -1
			}
			return 0
		})
		return neighbors.slice(0, k);
	}

	// _computeDistances computes the partial dot products and l2 distances of an embedding
	// from all the centres
	_computeDistances(embedding) {
		var subdims = this.codewords.shape[0];
		var nCenters = this.codewords.shape[1];
		var distances = [];
		for (var i = 0; i < subdims; i++) {
			var codeword = this.codewords.pick(i, embedding.get(i));
			var centers = this.codewords.pick(i);
			var dotProducts = np.dot(codeword, centers.T);
			var squareSums = np.dot(centers, centers.T).diag();
			var stacked = np.stack([dotProducts, squareSums], -1);
			distances.push(stacked);
		}

		return np.stack(distances);
	}
}

export const loadModel = async function(url) {
	const model = await utils.fetchModel(url);
	var embeddings = utils.unpackVectors(model.embeddings, 'uint16');
	var codewords = utils.unpackVectors(model.codewords, 'float64');
	return new WordEmbeddings(embeddings, codewords, model.vocabulary);
};