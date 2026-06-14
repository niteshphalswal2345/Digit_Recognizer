import os
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras import layers, models
from PIL import Image
import io

# --- 1. Model Preparation ---
MODEL_PATH = 'mnist_cnn.h5'

def train_model():
    print("Training CNN model on MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocessing
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    print("Model saved as mnist_cnn.h5")
    return model

# Load or Train
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = train_model()

# --- 2. Flask App ---
app = Flask(__name__)

# UI Design (Based on your Neo-Brutalist & Terminal style)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Digit Recognizer | Neo-Brutalist</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f0f0f0;
            --primary: #ffd93d;
            --secondary: #66d9ef;
            --accent: #ff6b6b;
            --dark: #000000;
            --border: 4px solid #000;
            --shadow: 8px 8px 0px #000;
        }

        body {
            background-color: var(--bg);
            font-family: 'Space Grotesk', sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
            margin: 0;
        }

        .header {
            background: var(--secondary);
            border: var(--border);
            box-shadow: var(--shadow);
            padding: 20px 40px;
            margin-bottom: 40px;
            transform: rotate(-1deg);
        }

        h1 { margin: 0; text-transform: uppercase; letter-spacing: -1px; }

        .main-container {
            display: flex;
            gap: 40px;
            flex-wrap: wrap;
            justify-content: center;
        }

        /* Canvas Area */
        .canvas-box {
            background: white;
            border: var(--border);
            box-shadow: var(--shadow);
            padding: 20px;
            position: relative;
        }

        #canvas {
            background: #000;
            cursor: crosshair;
            border: 2px solid #000;
        }

        .controls {
            margin-top: 20px;
            display: flex;
            gap: 15px;
        }

        button {
            font-family: 'Space Mono', monospace;
            font-weight: bold;
            padding: 12px 24px;
            border: var(--border);
            cursor: pointer;
            box-shadow: 4px 4px 0px #000;
            transition: 0.1s;
        }

        button:active { transform: translate(2px, 2px); box-shadow: 2px 2px 0px #000; }
        .btn-clear { background: var(--accent); }
        .btn-predict { background: var(--primary); }

        /* Terminal Area */
        .terminal {
            background: #1a1a1a;
            color: #98fb98;
            font-family: 'Space Mono', monospace;
            width: 350px;
            border: var(--border);
            box-shadow: var(--shadow);
            padding: 0;
            overflow: hidden;
        }

        .terminal-header {
            background: #333;
            padding: 8px 15px;
            display: flex;
            gap: 8px;
            border-bottom: 2px solid #000;
        }

        .dot { width: 12px; height: 12px; border-radius: 50%; border: 1px solid #000; }
        .red { background: #ff5f56; }
        .yellow { background: #ffbd2e; }
        .green { background: #27c93f; }

        .terminal-content {
            padding: 20px;
            min-height: 250px;
        }

        .result-val {
            font-size: 3rem;
            color: var(--primary);
            display: block;
            margin-top: 10px;
        }

        .highlight { color: var(--secondary); }
    </style>
</head>
<body>

    <div class="header">
        <h1>Handwritten Digit AI</h1>
    </div>

    <div class="main-container">
        <!-- Drawing Pad -->
        <div class="canvas-box">
            <canvas id="canvas" width="280" height="280"></canvas>
            <div class="controls">
                <button class="btn-clear" onclick="clearCanvas()">CLEAR [X]</button>
                <button class="btn-predict" onclick="predict()">PREDICT [RUN]</button>
            </div>
        </div>

        <!-- Terminal Output -->
        <div class="terminal">
            <div class="terminal-header">
                <div class="dot red"></div>
                <div class="dot yellow"></div>
                <div class="dot green"></div>
                <span style="color: #666; font-size: 12px; margin-left: 10px;">predict_digit.py</span>
            </div>
            <div class="terminal-content" id="output">
                <div>> System ready...</div>
                <div>> Draw a digit on the left.</div>
                <div id="prediction-result" style="margin-top: 20px; display: none;">
                    > Analysis complete.<br>
                    > Detected Digit: <span class="result-val" id="digit">?</span>
                    > Confidence: <span id="conf" class="highlight">0%</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let drawing = false;

        // Canvas Setup
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';

        canvas.addEventListener('mousedown', () => drawing = true);
        canvas.addEventListener('mouseup', () => {
            drawing = false;
            ctx.beginPath();
        });
        canvas.addEventListener('mousemove', draw);

        function draw(e) {
            if (!drawing) return;
            const rect = canvas.getBoundingClientRect();
            ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
        }

        function clearCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            document.getElementById('prediction-result').style.display = 'none';
        }

        async function predict() {
            const dataUrl = canvas.toDataURL('image/png');
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });
            
            const result = await response.json();
            
            if(result.success) {
                document.getElementById('prediction-result').style.display = 'block';
                document.getElementById('digit').innerText = result.digit;
                document.getElementById('conf').innerText = (result.confidence * 100).toFixed(2) + '%';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Open image and convert to grayscale
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        # Resize to 28x28
        img = img.resize((28, 28))
        img_array = np.array(img)
        
        # Normalize and Reshape for CNN
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # Predict
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        return jsonify({
            'success': True,
            'digit': int(digit),
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
