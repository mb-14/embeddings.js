var wordEmbeddings;
var sentimentCNN;

function init() {
	var modelStatus = document.getElementById("model-status");

	var sentimentClassification = {
		results: document.getElementById("sentiment-classification-results"),
		input: document.getElementById("sentiment-input")
	};


    var numWords = 0;

	sentimentClassification.input.addEventListener("input", () => {
	     // Convert to lower case and remove all punctuations.
	     const inputText = sentimentClassification.input.value
		   .trim()
		   .toLowerCase()
		   .replace(/(\.|\,|\!)/g, "")
		   .split(" ");

        if (numWords == inputText.length) {
            return;
        }

        numWords = inputText.length

		classifySentiment(inputText).then(result => {
		sentimentClassification.results.innerHTML =
			"Inference result (0 - negative; 1 - positive): " +
			result.score.toFixed(6) +
			" (elapsed: " +
			result.elapsed.toFixed(2) +
			" ms)";
		});
	});

	modelStatus.innerHTML = "Loading model...";
	loadModels().then(results => {
		modelStatus.innerHTML = "";
		wordEmbeddings = results.wordEmbeddings;
		sentimentLSTM = results.sentimentLSTM;
	});
}

async function classifySentiment(inputText) {
	const inputSequence = wordEmbeddings._transformSequence(inputText, 100);
	const beginMs = performance.now();
	const predictOut = sentimentLSTM.predict(inputSequence.expandDims(0));
	const score = predictOut.dataSync()[0];
	const elapsed = performance.now() - beginMs;
	return {score, elapsed}
}

async function loadModels() {
	await tf.setBackend('wasm');
	const sentimentLSTM = await tf.loadLayersModel("../../../assets/sentiment_lstm/model.json");
    const wordEmbeddings = await embeddings.loadModel("../../../assets/word-embeddings.json");
	return { sentimentLSTM, wordEmbeddings };
}

window.onload = init;
