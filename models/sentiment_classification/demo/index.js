function init() {
	var worker = new PromiseWorker(new Worker('worker.js'));

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

		worker.send("classifySentiment", inputText).then(result => {
		sentimentClassification.results.innerHTML =
			"Inference result (0 - negative; 1 - positive): " +
			result.score.toFixed(6) +
			" (elapsed: " +
			result.elapsed.toFixed(2) +
			" ms)";
		});
	});

	modelStatus.innerHTML = "Loading model...";
	worker.send("loadModels").then(() => {
		modelStatus.innerHTML = "";
	});

}



window.onload = init;
