function init() {
	var loaded = false;
	var worker = new Worker('worker.js');
	var resultsDiv = document.getElementById("results");
	var nearestNeighbours = document.getElementById("nearest-neighbors");
	worker.addEventListener('message', function(input) {
		var result = input.data;
		switch (result.type) {
			case 'load_model':
				loaded = result.data.loaded;
				nearestNeighbours.disabled = false;
				resultsDiv.innerHTML = "";
				break;
			case 'nearest_neighbors':
				resultsDiv.innerHTML = "";
				for (var i = 0; i< result.data.neighbors.length; i++) {
					resultsDiv.innerHTML  += i+1 + ". " + result.data.neighbors[i].word + "<br/>";
				}
				
		}
	});

	nearestNeighbours.addEventListener("click", () =>{
		if (loaded === false) {
			return;
		}
		var word = document.getElementById("word-input").value;
		resultsDiv.innerHTML = "Loading...";
		worker.postMessage({type: 'nearest_neighbors', data: {word: word}});
	});

	nearestNeighbours.disabled = true;
	resultsDiv.innerHTML = "Loading model...";
	worker.postMessage({type: 'load_model'});
}

window.onload = init;