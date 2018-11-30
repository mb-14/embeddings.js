const utils = require('./utils');
const model = require('./model');
const np = require('numjs');

function WordEmbeddings() {
	this.loadModel();
}

WordEmbeddings.prototype.loadModel = function() {
	var embeddings = utils.unpackVectors(model.embeddings, 'uint16');
	this.codewords = utils.unpackVectors(model.codewords, 'float64');
	var embeddingsMap = {};
	for (var i =0; i < model.vocabulary.length; i++) {
		embeddingsMap[model.vocabulary[i]] = embeddings.pick(i);
	}
	this.vocabulary = model.vocabulary;
	this.embeddingsMap = embeddingsMap;
};

WordEmbeddings.prototype._getVector = function(word) {
	var embedding = this.embeddingsMap[word];
	var vector = this.codewords.pick(0, embedding.get(0));
	for (var i = 1; i < embedding.shape[0]; i++) {
		var codeword = this.codewords.pick(i, embedding.get(i));
		vector = np.concatenate(vector, codeword);
	}
	return vector;
};

WordEmbeddings.prototype.getVector = function(word) {
	return this._getVector(word).toString();
};

WordEmbeddings.prototype.getCosineDistance = function(word1, word2) {
	var embedding1 = this.embeddingsMap[word1];
	var embedding2 = this.embeddingsMap[word2];
	var dotProduct = 0;
	var abs1 = 0;
	var abs2 = 0;
	for (var i = 0; i < embedding1.shape[0]; i++) {
		var c1 = this.codewords.pick(i, embedding1.get(i));
		var c2 = this.codewords.pick(i, embedding2.get(i));
		dotProduct += np.dot(c1, c2).get(0);
		abs1 += np.dot(c1, c1).get(0);
		abs2 += np.dot(c2, c2).get(0);
	}
	abs1 = Math.sqrt(abs1);
	abs2 = Math.sqrt(abs2);
	var cosineDistance = dotProduct/(abs1*abs2);
	return cosineDistance;
};

WordEmbeddings.prototype.getNearestNeighbors = function(word, count=5) {
	var neighbors = [];
	var embedding = this.embeddingsMap[word];
	var subdims = this.codewords.shape[0];
	var centers = this.codewords.shape[1];
	var abs = 0;
	var stacks = [];
	for (var i = 0; i < subdims; i++) {
		var codeword = this.codewords.pick(i, embedding.get(i));
		abs += np.dot(codeword, codeword).get(0);
		var centers = this.codewords.pick(i);
		var dotProducts = np.dot(codeword, centers.T);
		var squareSums = np.dot(centers, centers.T).diag();
		var stacked = np.stack([dotProducts, squareSums], -1);
		stacks.push(stacked);
	}
	abs = Math.sqrt(abs);
	var lookupTable = np.stack(stacks);
	for(var i = 0; i < this.vocabulary.length; i++) {
		var word1 = this.vocabulary[i];
		
		if (word1 === word) {
			continue;
		}
		
		var embedding1 = this.embeddingsMap[word1];
		var dotProduct = 0;
		var abs1 = 0;
		for (var j = 0; j < subdims; j++) {
			var center = embedding1.get(j);
			dotProduct += lookupTable.get(j, center, 0);
			abs1 += lookupTable.get(j, center, 1);
		}

		var nn = {
			distance: dotProduct/(abs*Math.sqrt(abs1)),
			word: word1,
			index: i
		};
		neighbors.push(nn);
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
	return neighbors.slice(0, count);
};

export const load = function() {
	return new WordEmbeddings();
}