from flask import Flask, request, jsonify

from topicdetection import load_data_and_calculate_topics

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/topics')
def topics():
    group = request.args.get('group')
    date_from = request.args.get('datefrom')
    date_to = request.args.get('dateto')
    topics = load_data_and_calculate_topics(group)
    return jsonify(topics=topics)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)