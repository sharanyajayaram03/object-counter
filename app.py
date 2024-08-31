from flask import Flask, jsonify, render_template, request
from flask_cors import CORS  # Import CORS for handling cross-origin requests
import subprocess

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Route to serve your HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Route to run your Python script
@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
        return jsonify({'output': result.stdout})  # Return JSON response with script output
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message with status code 500

if __name__ == '__main__':
    app.run(debug=True)  # Run Flask app in debug mode
