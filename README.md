# Handwritten Digit Recognizer

A full-stack AI web application that recognizes handwritten digits (0-9) drawn on a canvas using a CNN model trained on the MNIST dataset.

---

## Tech Stack

- **Frontend:** HTML, CSS, JavaScript, Canvas API
- **Backend:** Flask (Python)
- **ML Model:** CNN (Convolutional Neural Network) using TensorFlow
- **Dataset:** MNIST (70,000 handwritten digit images)
- **Accuracy:** 99%+

---

## Project Structure

```
digit-recognizer/
├── app.py
├── train_model.py
├── requirements.txt
├── runtime.txt
├── render.yaml
├── .gitignore
└── templates/
    └── index.html
```

---

## How It Works

When you draw a digit on the canvas:

1. The drawing is captured as a base64 PNG image
2. Converted to grayscale and inverted to match MNIST format
3. Cropped tightly around the digit
4. Resized to 28×28 pixels and centered
5. Normalized to values between 0 and 1
6. Sent to the Flask backend via a POST request
7. CNN model predicts the digit and returns confidence scores
8. Top 4 predictions are displayed with a confidence bar

---

## What is MNIST?

MNIST stands for Modified National Institute of Standards and Technology. It contains 70,000 real handwritten digit images collected from American Census Bureau employees and high school students in the 1980s–90s. Each image is 28×28 pixels in grayscale. It is considered the "Hello World" of machine learning.

---

## Why CNN Over SVM?

We initially built this with an SVM (Support Vector Machine) model but switched to CNN for these reasons:

| | SVM | CNN |
|---|---|---|
| Accuracy | ~97% | 99%+ |
| Understands shapes | No | Yes |
| Handles different styles | Poor | Much better |
| Training time | 5–15 min | 5–10 min |

CNN understands geometry — strokes, curves, corners — which makes it handle real hand-drawn digits much better than SVM which just compares raw pixel values.

---

## Complete Setup Guide From Scratch

### Prerequisites

- Python 3.11.x (TensorFlow does not support Python 3.13 or 3.14 yet)
- VS Code
- Git

### Step 1 — Install Python 3.11

Go to https://python.org/downloads/release/python-3119 and download Windows installer (64-bit). During installation check "Add Python to PATH". Verify with:

```
py -3.11 --version
```

### Step 2 — Create Project Folder

1. Open VS Code
2. Click File → Open Folder
3. Create a new folder called `digit-recognizer`
4. Click Select Folder

### Step 3 — Create File Structure

Create these files in VS Code sidebar:

```
digit-recognizer/
├── app.py
├── train_model.py
├── requirements.txt
└── templates/
    └── index.html
```

To create the templates folder, right click in the sidebar and click New Folder.

### Step 4 — Open Terminal in VS Code

Press Ctrl + ` to open the terminal. It opens directly inside your project folder.

### Step 5 — Create Virtual Environment With Python 3.11

```
py -3.11 -m venv venv311
```

### Step 6 — Activate Virtual Environment

```
venv311\Scripts\activate
```

You will see (venv311) in your terminal. Every time you reopen VS Code you need to run this activate command again.

### Step 7 — Install Libraries

```
pip install flask scikit-learn numpy pillow joblib pandas tensorflow gunicorn
```

### Step 8 — Train the Model

```
python train_model.py
```

This downloads the MNIST dataset, trains the CNN model, and saves model.keras in your project folder. Takes 5–10 minutes. You will see accuracy climbing each epoch and at the end:

```
✅ Test Accuracy: 99.4%
✅ Model saved as model.keras
```

You only need to run this once. model.keras is saved permanently.

### Step 9 — Run the Flask Server

```
python app.py
```

You will see:

```
✅ CNN Model loaded successfully.
* Running on http://127.0.0.1:5000
```

### Step 10 — Open the App

Open your browser and go to:

```
http://localhost:5000
```

Draw a digit and click Predict.

### Step 11 — Stop and Restart Later

To stop the server press Ctrl+C. Next time you want to run it:

```
venv311\Scripts\activate
python app.py
```

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| python not recognized | Reinstall Python and check Add to PATH |
| No module named tensorflow | Make sure venv311 is activated, not venv |
| msvcp140_1.dll not found | Install Microsoft C++ Redistributable from https://aka.ms/vs/17/release/vc_redist.x64.exe |
| model.keras not found | Run python train_model.py first |
| fetch_openml requires pandas | Run pip install pandas |
| venv311\Scripts\activate fails | Run Set-ExecutionPolicy RemoteSigned first |

---

## Tips for Best Prediction Accuracy

- Draw large, filling at least 60% of the canvas
- Draw in the center of the canvas
- Use thick brush strokes (slider to the right)
- Draw digits in standard style (avoid European style 7 with a curved tail)
- Draw slow and deliberate strokes
- Digits 0 and 8 are easiest, try those first to verify the model works

---

## Pushing to GitHub

### First Time Setup

```
git config --global user.name "Your Name"
git config --global user.email "youremail@gmail.com"
git init
git add .
git commit -m "Initial commit - Handwritten digit recognizer with CNN"
git branch -M main
git remote add origin https://github.com/YourUsername/digit-recognizer.git
git push -u origin main
```

### Every Time You Make Changes

```
git add .
git commit -m "describe what you changed"
git push
```

---

## Deploying on Render

1. Go to https://render.com and sign up with GitHub
2. Click New → Web Service
3. Connect your digit-recognizer repository
4. Set Build Command to:
```
pip install --upgrade pip && pip install -r requirements.txt && python train_model.py
```
5. Set Start Command to:
```
gunicorn app:app
```
6. Set Python Version environment variable to 3.11.6
7. Click Create Web Service
8. Wait 20–30 minutes for first deploy (installs TensorFlow and trains the model)
9. Your app will be live 

Note: Render free tier spins down after 15 minutes of inactivity. First visit after inactivity takes about 30 seconds to wake up.

---

## What I Learned Building This

- Setting up Python virtual environments and managing multiple Python versions
- Installing and managing libraries with pip
- How the MNIST dataset works and why it is the standard for digit recognition
- Difference between SVM and CNN and why CNN is better for image recognition
- How a CNN actually sees and recognizes images using pixel values
- Data augmentation to handle different handwriting styles
- Flask backend and REST APIs
- Connecting a frontend canvas to a Python ML model
- Fixing real-world deployment errors
- Git version control and GitHub
- Deploying a Flask app on Render

---

## Author

Kanneboina Akhila
GitHub: https://github.com/emailakhilak
