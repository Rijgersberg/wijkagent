from datetime import datetime

from flask import Flask, request, jsonify

from topicdetection import load_data_and_calculate_topics, get_docs_for_topic

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/topics', methods=['GET'])
def topics():
    group = request.args.get('group')
    date_from = request.args.get('datefrom')
    date_to = request.args.get('dateto')

    date_format = '%Y%m%d'
    if date_from is not None:
        date_from = datetime.strptime(date_from, date_format)
    if date_to is not None:
        date_to = datetime.strptime(date_to, date_format)

    topics = load_data_and_calculate_topics(group, date_from, date_to)
    return jsonify(topics)


@app.route('/chats', methods=['GET'])
def chats():
    topic = int(request.args.get('topic'))
    modelid = request.args.get('modelid')

    return jsonify(get_docs_for_topic(modelid, topic))
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)