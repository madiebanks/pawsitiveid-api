from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
from keras.saving import register_keras_serializable
import numpy as np
from PIL import Image
import io
import os
import sys

app = Flask(__name__)
CORS(app)

# Register the custom wrapper to handle TF Hub layers
@register_keras_serializable()
class HubLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, hub_url, trainable=False, **kwargs):
        super().__init__(**kwargs)
        self.hub_url = hub_url
        self.trainable = trainable
        self.hub_layer = hub.KerasLayer(self.hub_url, trainable=self.trainable)

    def call(self, inputs):
        return self.hub_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hub_url": self.hub_url,
            "trainable": self.trainable
        })
        return config

# Load the trained .keras model
model = tf.keras.models.load_model(
    'breedModel2.keras',
    custom_objects={"HubLayerWrapper": HubLayerWrapper}
)

# ✅ Load breed label classes saved from training
try:
    breed_classes = np.load("breed_labels.npy", allow_pickle=True)
except FileNotFoundError:
    print("Error: 'breed_labels.npy' not found. Please make sure it exists in the project directory.")
    sys.exit(1)

# Image preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # Match input size for MobileNetV2
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# API route to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image'}), 400

    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = breed_classes[predicted_index]
    confidence = float(np.max(prediction))

    return jsonify({
        'breed': predicted_label,
        'confidence': round(confidence, 3)
    })

# Health check
@app.route('/', methods=['GET'])
def home():
    return "DogBreed Prediction API is running."

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)