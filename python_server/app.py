from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app, methods=['POST'], headers=['Content-Type', 'Accept'])

json_path = 'my-chatbot\python_server\data.json'

with open(json_path, 'r') as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intens"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tags"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns)
vocab_size = len(tokenizer.word_index) + 1

X = tokenizer.texts_to_sequences(patterns)
X_padded = pad_sequences(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(tags)

# Check if the model file already exists
model_file = 'my-chatbot\python_server\chatbot_model.h5'
if os.path.exists(model_file):
    # Load the pre-trained model
    model = load_model(model_file)

def get_response(user_input):
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=X_padded.shape[1])
    predicted_probabilities = model.predict(user_input_padded)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)
    predicted_tag = label_encoder.inverse_transform(predicted_class)

    for intent in data["intens"]:
        if intent["tags"] == predicted_tag:
            return intent["responses"]

@app.route('/', methods=['OPTIONS'])
def handle_options():
    return jsonify({'status': 'success'}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['message']

    bot_response = get_response(user_input)

    return jsonify({'botResponse': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
