import * as lzstring  from 'lz-string';
import * as np from 'numjs';

export const unpackVectors = function(data, type) {
	var jsonData = JSON.parse(lzstring.decompressFromBase64(data));
	var npArray = np.array(jsonData.vectors, type);
	npArray = np.NdArray.prototype.reshape.apply(npArray, jsonData.shape);
	return npArray;
}

export const fetchModel = async function(url) {
	var response = await fetch(url);
	var data = await response.json();
	return data;
}