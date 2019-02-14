var wordEmbeddings;

if ('function' === typeof importScripts) {
	self.addEventListener('message', input => {
		var event = input.data;
		importScripts('assets/embeddings.js');
		var result = {
			type: event.type,
			data: {}
		};
		switch(event.type) {
			case 'load_model':
			embeddings.loadModel("assets/model.json").then(w => {
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
}