function init() {
	var worker = new Worker('worker.js');
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
	worker.addEventListener('message', function(input) {
		var result = input.data;
		switch (result.type) {
			case 'load_model':
				nearestNeighbors.button.disabled = false;
				wordAnalogy.button.disabled = false;
				modelStatus.innerHTML = "";
				break;
			case 'nearest_neighbors':
				nearestNeighbors.results.innerHTML = "";
				for (var i = 0; i< result.data.neighbors.length; i++) {
					nearestNeighbors.results.innerHTML  += i+1 + ". " + result.data.neighbors[i].word + "<br/>";
				}
				break;
			case 'word_analogy':
				wordAnalogy.results.innerHTML = "";
				for (var i = 0; i< result.data.neighbors.length; i++) {
					wordAnalogy.results.innerHTML  += i+1 + ". " + result.data.neighbors[i].word + "<br/>";
				}
		}
	});

	nearestNeighbors.button.addEventListener("click", () =>{
		var word = nearestNeighbors.word.value;
		nearestNeighbors.results.innerHTML = "Loading...";
		worker.postMessage({type: 'nearest_neighbors', data: {word: word}});
	});

	wordAnalogy.button.addEventListener("click", () =>{
		var word1 = wordAnalogy.word1.value;
		var word2 = wordAnalogy.word2.value;
		var word3 = wordAnalogy.word3.value;
		wordAnalogy.results.innerHTML = "Loading...";
		worker.postMessage({type: 'word_analogy', data: {word1: word1, word2: word2, word3: word3}});
	});

	nearestNeighbors.button.disabled = true;
	wordAnalogy.button.disabled = true;
	modelStatus.innerHTML = "Loading model...";
	worker.postMessage({type: 'load_model'});
}

window.onload = init;