import joblib
import yaml
import sys
import os
# Calculate the project root.
# __file__ is /home/afif/Documents/coding/python/projects/twitter_sentiment_analysis/api/app.py
# os.path.abspath(__file__) -> absolute path
# os.path.dirname(os.path.abspath(__file__)) -> /home/afif/Documents/coding/python/projects/twitter_sentiment_analysis/api
# os.path.dirname(os.path.dirname(os.path.abspath(__file__))) -> /home/afif/Documents/coding/python/projects/twitter_sentiment_analysis/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the Python path as the first entry
sys.path.insert(0, project_root)
from src.data_preprocessing import clean_text
from textblob import TextBlob
import numpy as np
from scipy.sparse import hstack
from flask import Flask, request, jsonify
from flask_cors import CORS


with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

app = Flask(__name__)
CORS(app)
model = joblib.load(config['model']['model_path'])
vectorizer = joblib.load(config['model']['vectorizer_path'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_text = data['text']
        processed_text = clean_text(input_text)
        X_text = vectorizer.transform([processed_text])
        
        polarity_score = TextBlob(input_text).sentiment.polarity
        X_polarity = np.array([[polarity_score]])

        X_combined = hstack([X_text, X_polarity])

        prediction = model.predict(X_combined)[0]

        sentiment = 'positive' if prediction == 1 else 'negative'

        response = {
            'text': input_text,
            'sentiment': sentiment
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)