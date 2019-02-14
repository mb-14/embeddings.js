import * as lzstring  from 'lz-string';
import * as tf from '@tensorflow/tfjs';

export const unpackVectors = function(data, type) {
	var jsonData = JSON.parse(lzstring.decompressFromBase64(data));
	var array = tf.tensor(jsonData.vectors, jsonData.shape, type);
	return array;
}

export const fetchModel = async function(url) {
	var response = await fetch(url);
	var data = await response.json();
	return data;
}