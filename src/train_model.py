import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import yaml
import logging
import os
import matplotlib.pyplot as plt
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import confusion_matrix

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# logging
logging.basicConfig(filename=config['logging']['file'], level=config['logging']['level'],\
                    format='%(asctime)s - %(levelname)s - %(message)s')


def train_model():
    try:

        mlflow.set_experiment('sentiment_analysis')

        # load data
        df = pd.read_csv(config['data']['processed_path'])
        print(df['sentiment'].value_counts(normalize=True))
        df['polarity'] = pd.to_numeric(df['polarity'], errors='coerce')
        df = df.dropna(subset=['cleaned_text', 'polarity'])
        print('df shape: ', df.shape)
        vectorizer = TfidfVectorizer(max_features=config['model']['max_features'], ngram_range=(1, 2))

        # Transform 'cleaned_text' using TF-IDF
        X_text = vectorizer.fit_transform(df['cleaned_text'])

        # Get the 'polarity' feature and reshape it for horizontal stacking
        X_polarity = df['polarity'].values.reshape(-1, 1)

        # Combine TF-IDF features with 'polarity' feature
        X = hstack([X_text, X_polarity]) # Use hstack to combine horizontally
        X = X.tocsr() # Convert to Compressed Sparse Row format
        
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['model']['test_size'], random_state=config['model']['random_state'])

        # inisialisasi model
        lr_model = LogisticRegression(max_iter=1000)

        with mlflow.start_run():
            lr_model.fit(X_train, y_train)

            # evaluasi model
            y_pred = lr_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

             # log metrics kedalam MLflow
            mlflow.log_metric('accuracy', accuracy)
            mlflow.sklearn.log_model(lr_model, 'model')
            logging.info(f'Classification report:\n {report}')

            # visualisasi confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6,4))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            mlflow.log_artifact('confusion_matrix.png')
            
            # save model
            os.makedirs(os.path.dirname(config['model']['model_path']), exist_ok=True)
            joblib.dump(lr_model, config['model']['model_path'])
            joblib.dump(vectorizer, config['model']['vectorizer_path'])
            logging.info('Model and vectorizer saved successfully')
            mlflow.sklearn.log_model(lr_model, 'final_model') # Log final model to MLflow
            logging.info('Final model and vectorizer saved successfully')
            return accuracy
    except Exception as e:
        logging.error(f'Error in train model: {e}')
        raise

if __name__ == '__main__':
    acc = train_model()
    logging.info(f'Accuracy: {acc:.2f}')
    print(f'Accuracy: {acc:.2f}')
