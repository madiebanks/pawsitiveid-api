from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

#Load the trained model
model = tf.keras.models.load_model('breedModel.h5')

#Load label classes
label_encoder = LabelEncoder()
#Actual breed label list
breed_classes = []
label_encoder.fit(breed_classes)

#Image preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    return jsonify({'breed': predicted_label,
                    'confidence': round(confidence, 3)})

@app.rout('/', methods=['GET'])
def home():
    return "DogBreed Prediction API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)