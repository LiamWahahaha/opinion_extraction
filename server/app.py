from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
import util.personal_opinion as personal_opinion

# configuration
DEBUG = True

app = Flask(__name__)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/extract-opinion', methods=['POST'])
def extract_opinion():
    req_data = request.get_json()
    result = personal_opinion.get_opinions(req_data['msg'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)