from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#Load the trained model
model = tf.keras.models.load_model('20250615_171214_full_image_set_mobilenetv2_Adam.h5')

#Load label classes
label_encoder = LabelEncoder()
#Actual breed label list
breed_classes = ["Afghan Hound", "African Wild Dog", "Airedale Terrier", "American Staffordshire Terrier", "Appenzeller Sennenhund", "Australian Terrier", "Bedlington Terrier", 
                 "Bernese Mountain Dog", "Blenheim Cavalier King Charles Spaniel", "Border Collie", "Border Terrier", "Boston Terrier", "Bouvier Des Flandres", "Brussels Griffon", 
                 "Brittany", "Cardigan Welsh Corgi", "Chesapeake Bay Retriever", "Chihuahua", "Dandie Dinmont Terrier", "Doberman Pinscher", "English Setter", "English Springer Spaniel", 
                 "Entlebucher Mountain Dog", "American Eskimo Dog", "German Shepherd", "German Shorthaired Pointer", "Gordon Setter", "Great Dane", "Great Pyrenees", 
                 "Greater Swiss Mountain Dog", "Ibizan Hound", "Irish Setter", "Irish Terrier", "Irish Water Spaniel", "Irish Wolfhound", "Italian Greyhound", "Japanese Chin", 
                 "Kerry Blue Terrier", "Labrador Retriever", "Lakeland Terrier", "Leonberger", "Lhasa Apso", "Maltese", "Xoloitzcuintli", "Newfoundland", 
                 "Norfolk Terrier", "Norwegian Elkhound", "Norwich Terrier", "Old English Sheepdog", "Pekingese", "Pembroke Welsh Corgi", "Pomeranian", "Rhodesian Ridgeback", 
                 "Rottweiler", "Saint Bernard", "Saluki", "Samoyed", "Scottish Terrier", "Scottish Deerhound", "Sealyham Terrier", "Shih Tzu", "Siberian Husky", 
                 "Staffordshire Bull Terrier", "Sussex Spanel", "Tibetan Terrier", "Treeing Walker Coonhound", "Weimaraner", "Welsh Springer Spaniel", "West Highland White Terrier", 
                 "Yorkshire Terrier", "Affenpinscher", "Basenji", "Basset Hound", "Beagle", "Black and Tan Coonhound", "Bloodhound", "Bluetick Coonhound", "Borzoi", "Boxer", "Briard", 
                 "Bullmastiff", "Cairn Terrier", "Chow Chow", "Clumber Spaniel", "Cocker Spaniel", "Collie", "Curly-Coated Retriever", "Dhole", "Carolina Dog", "Flat-Coated Retriever", 
                 "Giant Schnauzer", "Golden Retriever", "Belgian Sheepdog", "Keeshond", "Australian Kelpie", "Komondor", "Kuvasz", "Alaskan Malamute", "Belgian Malinois", "Miniature Pinshcer", 
                 "Poodle (Miniature)", "Miniature Schnauzer", "Papillon", "Pug", "Redbone Coonhound", "Schipperke", "Silky Terrier", "Soft Coated Wheaten Terrier", 
                 "Poodle (Standard)", "Standard Schnauzer", "Poodle (Toy)", "Toy Fox Terrier", "Vizsla", "Whippet", "Wire Fox Terrier"]
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
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File must be an image'}), 400
    
    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)

    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    return jsonify({'breed': predicted_label,
                    'confidence': round(confidence, 3)})

@app.route('/', methods=['GET'])
def home():
    return "DogBreed Prediction API is running."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)