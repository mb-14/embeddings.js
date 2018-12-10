function init() {
	var loaded = false;
	var worker = new Worker('worker.js');
	var resultsDiv = document.getElementById("results");

	worker.addEventListener('message', function(input) {
		var result = input.data;
		switch (result.type) {
			case 'load_model':
				loaded = result.data.loaded;
				console.log("Loaded");
				break;
			case 'nearest_neighbors':
				resultsDiv.innerHTML = "";
				for (var i = 0; i< result.data.neighbors.length; i++) {
					resultsDiv.innerHTML  += result.data.neighbors[i].word + "<br/>";
				}
				
		}
	});

	document.getElementById("nearest-neighbors").addEventListener("click", () =>{
		if (loaded === false) {
			return;
		}
		var word = document.getElementById("word-input").value;
		resultsDiv.innerHTML = "Loading...";
		worker.postMessage({type: 'nearest_neighbors', data: {word: word}});
	});

	worker.postMessage({type: 'load_model'});
}

window.onload = init;