const lzstring = require('lz-string');
const np = require('numjs');



module.exports.unpackVectors = function(data, type) {
	var jsonData = JSON.parse(lzstring.decompressFromBase64(data));
	var npArray = np.array(jsonData.vectors, type);
	npArray = np.NdArray.prototype.reshape.apply(npArray, jsonData.shape);
	return npArray;
}

module.exports.magnitude = function(vector) {
	l2Squared = np.dot(vector, vector);
	return Math.sqrt(l2Squared);
}
