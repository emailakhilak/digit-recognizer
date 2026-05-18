"""
Flask backend for Handwritten Digit Recognizer (CNN version).
Run: python app.py
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import base64
import io
import os

app = Flask(__name__)

# ── Load CNN model ─────────────────────────────────────────────
MODEL_PATH = "model.keras"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.keras not found. Please run: python train_model.py"
    )

import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ CNN Model loaded successfully.")


def preprocess_image(image_data_url: str) -> np.ndarray:
    """
    Convert base64 canvas PNG → 28×28 grayscale numpy array
    matching MNIST format (white digit on black background, normalized 0–1).
    """
    header, encoded = image_data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    # Flatten transparency onto white background
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    img = background.convert("L")  # grayscale

    arr = np.array(img, dtype=np.float32)

    # Auto-invert: MNIST needs white digit on black background
    if arr.mean() > 127:
        arr = 255.0 - arr

    # Crop tightly around digit
    thresh = (arr > 30).astype(np.uint8)
    rows = np.any(thresh, axis=1)
    cols = np.any(thresh, axis=0)

    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        pad = 20
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1] - 1, cmax + pad)

        arr = arr[rmin:rmax+1, cmin:cmax+1]

    # Resize digit to 20×20 and center in 28×28 (matches MNIST convention)
    digit_img = Image.fromarray(arr.astype(np.uint8))
    digit_img = digit_img.resize((20, 20), Image.LANCZOS)

    final = Image.new("L", (28, 28), 0)
    final.paste(digit_img, (4, 4))

    # Normalize to 0–1 and reshape for CNN input (1, 28, 28, 1)
    pixels = np.array(final, dtype=np.float32) / 255.0
    pixels = pixels.reshape(1, 28, 28, 1)
    return pixels


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        pixels = preprocess_image(data["image"])

        # CNN outputs softmax probabilities for all 10 digits
        probs = model.predict(pixels, verbose=0)[0]  # shape (10,)

        digit_probs = sorted(
            enumerate(probs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        predicted_digit = digit_probs[0][0]
        confidence = round(digit_probs[0][1] * 100, 1)

        top_predictions = [
            {"digit": d, "confidence": round(c * 100, 1)}
            for d, c in digit_probs[:4]
            if c * 100 > 0.05
        ]

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "top_predictions": top_predictions,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)