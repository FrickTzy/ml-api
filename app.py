from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/')
def home():
    return jsonify({'message': 'Hello from your Flask API on Render!'})


@app.route('/add', methods=['GET'])
def add():
    try:
        a = float(request.args.get('a', 0))
        b = float(request.args.get('b', 0))
        return jsonify({'result': a + b})
    except ValueError:
        return jsonify({'error': 'Invalid input'}), 400


@app.route('/echo', methods=['POST'])
def echo():
    data = request.json
    return jsonify({'you_sent': data})


if __name__ == '__main__':
    app.run(debug=True)
