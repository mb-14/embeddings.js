function init() {
	var wordEmbeddings;
	var sentimentCNN;
	var modelStatus = document.getElementById("model-status");
	var nearestNeighbors = {
		button: document.getElementById("nearest-neighbors"),
		results: document.getElementById("results"),
		word: document.getElementById("word-input")
	};
	var wordAnalogy = {
		button: document.getElementById("word-analogy"),
		results: document.getElementById("word-analogy-results"),
		word1: document.getElementById("word1-input"),
		word2: document.getElementById("word2-input"),
		word3: document.getElementById("word3-input")
	};

	var sentimentClassification = {
		results: document.getElementById("sentiment-classification-results"),
		input: document.getElementById("sentiment-input")
	};

	nearestNeighbors.button.addEventListener("click", () => {
		var word = nearestNeighbors.word.value;
		nearestNeighbors.results.innerHTML = "Loading...";
		wordEmbeddings.getNearestNeighbors(word).then(results => {
			nearestNeighbors.results.innerHTML = "";
			for (var i = 0; i < results.length; i++) {
				nearestNeighbors.results.innerHTML +=
					i + 1 + ". " + results[i].word + "<br/>";
			}
		});
	});

	wordAnalogy.button.addEventListener("click", () => {
		var word1 = wordAnalogy.word1.value;
		var word2 = wordAnalogy.word2.value;
		var word3 = wordAnalogy.word3.value;
		wordAnalogy.results.innerHTML = "Loading...";
		wordEmbeddings.wordAnalogy(word1, word2, word3).then(results => {
			wordAnalogy.results.innerHTML = "";
			for (var i = 0; i < results.length; i++) {
				wordAnalogy.results.innerHTML +=
					i + 1 + ". " + results[i].word + "<br/>";
			}
		});
	});

	sentimentClassification.input.addEventListener("input", () => {
		// Convert to lower case and remove all punctuations.
		const text = sentimentClassification.input.value;
		const inputText = text
			.trim()
			.toLowerCase()
			.replace(/(\.|\,|\!)/g, "")
			.split(" ");
		console.log(inputText);
		const inputSequence = wordEmbeddings._transformSequence(inputText, 200);
		inputSequence.print();
		const beginMs = performance.now();
		const predictOut = sentimentCNN.predict(inputSequence.expandDims(0));
		const score = predictOut.dataSync()[0];
		predictOut.dispose();
		inputSequence.dispose();
		const elapsed = performance.now() - beginMs;
		sentimentClassification.results.innerHTML =
			"Inference result (0 - negative; 1 - positive): " +
			score.toFixed(6) +
			" (elapsed: " +
			elapsed.toFixed(2) +
			" ms)";
	});

	nearestNeighbors.button.disabled = true;
	wordAnalogy.button.disabled = true;
	modelStatus.innerHTML = "Loading model...";
	loadModels().then(results => {
		nearestNeighbors.button.disabled = false;
		wordAnalogy.button.disabled = false;
		modelStatus.innerHTML = "";
		wordEmbeddings = results.wordEmbeddings;
		sentimentCNN = results.sentimentCNN;
		window.wordEmbeddings = wordEmbeddings;
	});
}

async function loadModels() {
    let sentimentCNN;
	// const sentimentCNN = await tf.loadModel("assets/sentiment_cnn/model.json");
    const wordEmbeddings = await embeddings.loadModel("../../../pretrained/word-embeddings.json");
	return { sentimentCNN, wordEmbeddings };
}

window.onload = init;
