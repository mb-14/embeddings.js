class PromiseWorker {
	constructor(worker) {
		this.worker = worker;
		this.callbackMap = {
			log: (message) => {console.log(message)}
		};
		this.worker.onmessage = this.onmessage.bind(this);
	}

	onmessage(event) {
		var data = event.data;
		var id = data.id;
		var result = data.result;
		var callback = this.callbackMap[id];
		delete this.callbackMap[id];
		callback(result);
	}

	send(action, ...args) {
		var id = makeid(5);
		this.worker.postMessage({action:action, args: args, id: id});
		return new Promise((resolve, reject) => {
			this.callbackMap[id] = result => {
				resolve(result);
			}
		});
	}
}

function makeid(length) {
  var text = "";
  var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

  for (var i = 0; i < length; i++)
    text += possible.charAt(Math.floor(Math.random() * possible.length));

  return text;
}