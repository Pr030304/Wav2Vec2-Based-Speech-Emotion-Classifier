from flask import Flask, request, jsonify
import os
import tempfile
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from pred_entropy import inference_function
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for all routes

# Load the processor and model from the checkpoint directory
model_new = Wav2Vec2ForSequenceClassification.from_pretrained("checkpoint-5250", output_hidden_states=True)
processor_new = Wav2Vec2Processor.from_pretrained("checkpoint-processor")

# Inverse mapping for labels
inverse_label_map = {0: 'surprise', 1: 'angry', 2: 'neutral', 3: 'sad', 4: 'happy'}

cluster = []
cr = 0

@app.route('/')
def index():
    return "API is running"

@app.route('/predict', methods=['POST'])
def predict():
    global cluster
    global cr
    # Check if a file was sent in the POST request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Create a temporary file using mkstemp (safer on Windows)
    fd, temp_path = tempfile.mkstemp(suffix=".wav")
    try:
        # Write the uploaded file's contents to the temporary file
        with os.fdopen(fd, 'wb') as tmp_file:
            tmp_file.write(file.read())

        # Call your inference function (which should return a prediction index and probability list)
        cluster, predictions, probabilities = inference_function(model_new, processor_new, temp_path, clusters=cluster)


        # Return the result as JSON
        if predictions.startswith("unknown_type_"):
            if cr < len(cluster):
                s = "Uh Oh. This is not present in the dataset. This is a new emotion : "
                s += predictions
                predictions = s
            else:
                predictions = "Oh, this is not in the dataset but we have already seen this type of audio before : " + predictions

        cr = len(cluster)
        
        return jsonify({
            'prediction': predictions,
            'probabilities': probabilities
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)