import joblib
import yaml
import logging
import data_preprocessing
import numpy as np
from scipy.sparse import hstack
from textblob import TextBlob

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# logging
logging.basicConfig(filename=config['logging']['file'], level=config['logging']['level'],\
                    format='%(asctime)s - %(levelname)s - %(message)s')

def predict(text):
    try:
        model = joblib.load(config['model']['model_path'])
        vectorizer = joblib.load(config['model']['vectorizer_path'])

        original_text = text
        processed_text = data_preprocessing.clean_text(original_text)
        X_text = vectorizer.transform([processed_text])
        
        polarity_score = TextBlob(original_text).sentiment.polarity
        X_polarity = np.array([[polarity_score]])

        X_combined = hstack([X_text, X_polarity])

        logging.info(f'Shape of combined features (X_combined): {X_combined.shape}')

        prediction = model.predict(X_combined)[0]
        return prediction
    except Exception as e:
        logging.error(f'Error in prediction:{e}')
        raise

if __name__ == '__main__':
    sample_comment_bad = 'These feature is so bad'
    sample_comment_good = 'I love this product, it is amazing!'
    sample_comment_neutral = 'The weather is okay today.'

    print(f"Sample 1: '{sample_comment_bad}'")
    pred_bad = predict(sample_comment_bad)
    print(f"Prediction: {'Positive' if pred_bad == 1 else 'Negative'}\n")

    print(f"Sample 2: '{sample_comment_good}'")
    pred_good = predict(sample_comment_good)
    print(f"Prediction: {'Positive' if pred_good == 1 else 'Negative'}\n")

    print(f"Sample 3: '{sample_comment_neutral}'")
    pred_neutral = predict(sample_comment_neutral)
    print(f"Prediction: {'Positive' if pred_neutral == 1 else 'Negative'}\n")
