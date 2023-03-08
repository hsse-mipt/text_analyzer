import json
from flask import Flask, request, make_response as response

app = Flask(__name__)


@app.route("/")
def index():
    text = request.args.get('text')
    json_type = 'application/json'
    json_accepted = json_type in request.headers.get('Accept', '')
    if text:
        words = text.split()
        # making a prediction
        prediction = 0
        if json_accepted:
            res = response(json.dumps({'text': prediction}), 200)
        else:
            res = response(prediction, 200)
    else:
        res = response('text not found', 501)
    res.headers['Content-Type'] = json_type if json_accepted else 'text/plain'
    return res


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
