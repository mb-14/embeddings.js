var wordEmbeddings;

self.addEventListener('message', input => {
	var event = input.data;
	self.importScripts('dist/embeddings.bundle.js');
	var result = {
		type: event.type,
		data: {}
	}
	switch(event.type) {
		case 'load_model':
			embeddings.loadModel("dist/model.json").then(w => {
				wordEmbeddings = w;
				result.data = { loaded: true };
				self.postMessage(result);
			});
			break;
		case 'nearest_neighbors':
			var neighbors = wordEmbeddings.getNearestNeighbors(event.data.word);
			result.data = { neighbors: neighbors };
			self.postMessage(result);
			break;
	}
}); 