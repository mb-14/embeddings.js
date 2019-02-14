import * as utils from './utils';
import * as tf from '@tensorflow/tfjs';

class WordEmbeddings {

	constructor(codes, centroids, vocabulary) {
		this.vocabulary = vocabulary;
		this.centroids = centroids;
		this.codes = codes;
	}

	// _getVector returns the vector representation of a word as a tensor
	_getVector(word) {
		var index = this.vocabulary.indexOf(word);
		var codes = this._getSearchVector(index)
		var indices = tf.range(0, this.codes.shape[1], 1, 'int32');
		var search = tf.stack([indices, codes], -1);
		var vector = tf.gatherND(this.centroids, search).flatten();
		return vector;
	}

	_getSearchVector(index) {
		return this.codes.gather([index]).as1D();
	}

	// _getVector returns a Promise the vector representation of a word as a float array
	getVector(word) {
		return this._getVector(word).dataSync();
	}

	// getCosineDistance returns the cosine distance between two word vectors
	getCosineDistance(word1, word2) {
		var vec1 = this._getVector(word1);
		var vec2 = this._getVector(word2);
		var dotProduct = vec1.dot(vec2).asScalar();
		var abs1 = vec1.norm(2);
		var abs2 = vec2.norm(2);
		var cosineDistance = dotProduct.div(abs1).div(abs2);
		return cosineDistance.dataSync()[0];
	}

	// getNearestNeighbors returns the closest k word vectors from a given word vector 
	getNearestNeighbors(word, k=5) {
		var neighbors = tf.tensor1d([]);
		var vector = this._getVector(word);
		var abs = vector.norm(2).asScalar();
		// Precompute distances
		console.time("precompute_distances");
		var lookupTable = this._computeDistances(vector);
		console.timeEnd("precompute_distances");

		// Calculate distance for each word vector
		console.time("calulate");
		var subdims = this.centroids.shape[0];
		const searchIndices = tf.range(0, subdims, 1, 'int32').tile([this.vocabulary.length]);
		const searchVectors = this.codes.flatten();
		var search = tf.stack([searchIndices, searchVectors], -1);
		var dotProducts = tf.gatherND(lookupTable[0], search).reshape([this.vocabulary.length, -1]);
		var abs1 = tf.gatherND(lookupTable[1], search).reshape([this.vocabulary.length, -1]);
		dotProducts = dotProducts.sum([1]);
		abs1 = abs1.sum([1]);
		neighbors = dotProducts.div(abs.mul(abs1.sqrt()));
		console.timeEnd("calulate");
		
		// Get top K distances
		console.time("topk");
		var {values, indices} = tf.topk(neighbors, k+1);
		console.timeEnd("topk");
		values = values.dataSync();
		indices = indices.dataSync();
		var nearestNeighbors = [];
		for (var i = 1; i < indices.length; i++) {
			nearestNeighbors.push({
				word: this.vocabulary[indices[i]],
				distance: values[i]
			});
		}
		return nearestNeighbors;
	}

	// _computeDistances computes the partial dot products and l2 distances of an embedding
	// from all the centres
	_computeDistances(vector) {
		var subdims = this.centroids.shape[0];
		vector = vector.reshape([subdims, -1]);
		var squareSums = this.centroids.norm(2, 2).square();
		var dotProducts = [];
		for (var i = 0; i < subdims; i++) {
			var codeword = vector.gather([i]).squeeze();
			var centers = this.centroids.gather([i]).squeeze();
			var dotProduct = codeword.dot(centers.transpose());
			dotProducts.push(dotProduct);
		}
		dotProducts = tf.stack(dotProducts);
		return [dotProducts, squareSums];
	}
}

export const loadModel = async function(url) {
	const model = await utils.fetchModel(url);
	console.log("Unpacking codes");
	var codes = utils.unpackVectors(model.codes, 'int32');
	console.log("Unpacking centroids");
	var centroids = utils.unpackVectors(model.centroids, 'float32');
	return new WordEmbeddings(codes, centroids, model.vocabulary);
};