# simple_web_demo.py - Simple Web Demo with One Text Box
# Run with: py -3.11 simple_web_demo.py
# Open: http://localhost:5000

import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model

# Load model and tokenizer
print("Loading model...")
model = load_model("best_model.keras")
print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

id_to_word = {v: k for k, v in tokenizer.word_index.items()}
print("Ready! Open http://localhost:5000 in your browser\n")

# Simple HTML template
HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Swedish Word Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        input {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
            margin-bottom: 15px;
        }
        input:focus {
            outline: none;
            border-color: #4CAF50;
        }
        button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background-color: #45a049;
        }
        button:active {
            background-color: #3d8b40;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .error {
            color: #d32f2f;
            margin-top: 15px;
            padding: 10px;
            background-color: #ffebee;
            border-radius: 5px;
            display: none;
        }
        .prediction-item {
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 4px solid #4CAF50;
        }
        .rank {
            font-weight: bold;
            color: #4CAF50;
        }
        .word {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .prob {
            color: #666;
            font-size: 14px;
        }
        .input-echo {
            margin-top: 20px;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 5px;
            font-size: 14px;
            color: #333;
        }
        .spinner {
            text-align: center;
            color: #666;
            display: none;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🇸🇪 Swedish Next-Word Predictor</h1>
        <div class="subtitle">LSTM Neural Network on Swedish News Corpus</div>

        <label>Enter 5 Swedish words (separated by spaces):</label>
        <input type="text" id="words" placeholder="e.g. jag tycker om te och" />
        <button onclick="predict()">Predict Next Word</button>

        <div class="error" id="error"></div>
        <div class="spinner" id="spinner">⏳ Predicting...</div>

        <div class="results" id="results">
            <h3>Top 5 Predictions:</h3>
            <div id="predictions"></div>
            <div class="input-echo" id="inputEcho"></div>
        </div>
    </div>

    <script>
        // Allow Enter key to trigger predict
        document.getElementById('words').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                predict();
            }
        });

        async function predict() {
            const input = document.getElementById('words').value.trim().toLowerCase();
            const errorEl = document.getElementById('error');
            const resultsEl = document.getElementById('results');
            const spinnerEl = document.getElementById('spinner');
            const predictionsEl = document.getElementById('predictions');

            // Hide previous results
            errorEl.style.display = 'none';
            resultsEl.style.display = 'none';

            // Split and validate
            const words = input.split(/\s+/);
            if (words.length !== 5 || !input) {
                errorEl.textContent = `Error: You entered ${words.length} words, need exactly 5.`;
                errorEl.style.display = 'block';
                return;
            }

            // Show spinner
            spinnerEl.style.display = 'block';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ words: words })
                });

                const data = await response.json();
                spinnerEl.style.display = 'none';

                if (data.error) {
                    errorEl.textContent = 'Error: ' + data.error;
                    errorEl.style.display = 'block';
                    return;
                }

                // Show predictions
                predictionsEl.innerHTML = data.predictions.map((item, i) => `
                    <div class="prediction-item">
                        <span class="rank">#${i + 1}</span> 
                        <span class="word">${item.word}</span> 
                        <span class="prob">${item.prob.toFixed(2)}%</span>
                    </div>
                `).join('');

                document.getElementById('inputEcho').textContent = 
                    `Input: ${words.join(' ')} → ?`;

                resultsEl.style.display = 'block';

            } catch (error) {
                spinnerEl.style.display = 'none';
                errorEl.textContent = 'Server error. Is Flask running?';
                errorEl.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    words = data.get("words", [])

    if len(words) != 5:
        return jsonify({"error": f"Expected 5 words, got {len(words)}"})

    # Convert words to token IDs
    ids = [tokenizer.word_index.get(w.lower(), 0) for w in words]
    X = np.array([ids])

    # Get predictions
    probs = model.predict(X, verbose=0)[0]
    top5_idx = np.argsort(probs)[-5:][::-1]

    predictions = []
    for idx in top5_idx:
        word = id_to_word.get(idx, "<OOV>")
        prob = float(probs[idx]) * 100
        predictions.append({"word": word, "prob": prob})

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=False, port=5000)