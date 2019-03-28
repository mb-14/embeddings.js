importScripts('https://cdn.jsdelivr.net/npm/setimmediate@1.0.5/setImmediate.min.js')
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.10.3')
tf.setBackend('cpu')

importScripts('https://npmcdn.com/promise-worker/dist/promise-worker.register.js');
importScripts('../../../dist/embeddings.js');

var wordEmbeddings;
var sentimentCNN;

var api = {
    loadModels: function(respond){
        loadModelsAsync().then(result => {
        	wordEmbeddings = result.wordEmbeddings;
			sentimentLSTM = result.sentimentLSTM;
			respond();
        });
    },

    classifySentiment: function(respond, inputText) {
    	classifySentimentAsync(inputText).then(result => {
    		respond(result);
    	})
    }
};

onmessage = event => {
	var data = event.data;
    var respond = function(_response){
        postMessage({result: _response, id: data.id});
    };
    if (data.action && api[data.action]) {
        // prepend the params with the respond function
        var args = [respond].concat(data.args || []);
        api[data.action].apply(null, args);
    }
};


async function classifySentimentAsync(inputText) {
	const inputSequence = wordEmbeddings._transformSequence(inputText, 100);
	const beginMs = performance.now();
	const predictOut = sentimentLSTM.predict(inputSequence.expandDims(0));
	const score = predictOut.dataSync()[0];
	const elapsed = performance.now() - beginMs;
	return {score, elapsed}
}

async function loadModelsAsync() {
	const sentimentLSTM = await tf.loadModel("../../../pretrained/sentiment_lstm/model.json");
    const wordEmbeddings = await embeddings.loadModel("../../../pretrained/word-embeddings.json");
	return { sentimentLSTM, wordEmbeddings };
}