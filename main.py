from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
import time
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model when the application starts
model = None
model_path = "models/Resnet152.h5"

def load_model():
    try:
        global model
        # Load the model
        model = tf.keras.models.load_model(model_path, compile=False)  # Avoid unnecessary compilation

        # Optional: Display the model summary
        model.summary()
    except Exception as e:
        print(f"Error loading the model: {str(e)}")

# Preload the model
load_model()

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "photo" not in request.files:
        return jsonify({"error": "No photo provided"}), 400

    photo = request.files["photo"]
    if photo.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the photo with its original filename
        directory = "uploads"
        os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
        timestamp = int(time.time())
        timestamp_str = str(timestamp)
        photo_path = os.path.join(directory, timestamp_str + ".jpg")
        photo.save(photo_path)

        # Process the image
        img = Image.open(photo_path)
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0

        # Ensure the image has three channels (RGB)
        if img_array.shape[-1] == 1:
            img_array = tf.concat([img_array] * 3, axis=-1)

        # Make predictions
        class_labels = ['Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to', 'Canh chua', 'Cao lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo']
        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]
        return jsonify(predicted_label)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up: Delete the uploaded photo
        os.remove(photo_path)

if __name__ == "__main__":
    app.run(debug=True)
