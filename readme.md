# ✍️ AI Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) trained on the MNIST dataset that recognizes hand-drawn digits in real time. The application features a modern **Neo-Brutalist** design with a **Terminal-Style** interface.

![Output Sample](sample1.png)

## 🛠️ Features

* **Interactive Canvas:** Draw digits using your mouse.
* **AI Prediction:** TensorFlow-powered CNN model for high-accuracy predictions.
* **Terminal Interface:** Displays live logs and prediction results.
* **Automatic Training:** Trains the model automatically if no trained model file is found.

## 💻 Setup & Installation

### Windows

```cmd
python -m venv ai_env
ai_env\Scripts\activate
pip install flask tensorflow numpy pillow
python main.py
```

### Linux (Ubuntu/Kali)

```bash
python3 -m venv ai_env
source ai_env/bin/activate
pip install flask tensorflow numpy pillow
python main.py
```

## 🚀 Usage

1. Start the application.
2. Open `http://127.0.0.1:5000` in your browser.
3. Draw a digit (0–9) on the canvas.
4. Click **PREDICT**.
5. View the prediction result in the terminal panel.

