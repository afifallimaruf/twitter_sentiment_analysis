import pandas as pd
import nltk
import re
import yaml
import os
import logging
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import mark_negation
from textblob import TextBlob

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# load config.yaml
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# logging
logging.basicConfig(filename=config['logging']['file'], level=config['logging']['level'],\
                    format='%(asctime)s - %(levelname)s - %(message)s')
    
def clean_text(text):
    try:
        # Check if text is a string and not NaN
        if not isinstance(text, str) or pd.isna(text):
            return ''
        # ubah emoji ke bentuk teks
        text = emoji.demojize(text)
        text = re.sub(r'(\w)\1{2,}', r'\1', text)
        # ubah text ke bentuk lowercase
        text = text.lower()
        # hapus URLs, mentions, hashtags, dan karakter spesial
        text = re.sub(r'http\S+|www\.\S+|@\w+|#\w+|[^a-zA-Z\s]', '', text)
        stop_words = set(stopwords.words('english'))
        # membuat tokens
        tokens = word_tokenize(text)
        tokens = mark_negation(tokens)
        lemmatizer = WordNetLemmatizer()
        # hapus stopwords dan stem
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join(tokens)
        return text
    except Exception as e:
        logging.error(f'Error in clean text: {e}')
        raise
    

def load_data():
    try:
        # load data
        column_names = ['sentiment', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(os.path.join(config['data']['raw_path'], 'sentiment140.csv'), encoding='ISO-8859-1', names=column_names)
        logging.info('Data loaded successfully')
        return df    
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise

def preprocess_data():
    try:
        df = load_data()
        # Drop rows where 'text' is NaN
        df = df.dropna(subset=['text'])
        # Ensure 'text' column is of string type
        df['text'] = df['text'].astype(str)
        df['cleaned_text'] = df['text'].apply(clean_text)
        df = df.dropna(subset=['cleaned_text'])
        
        df[(df['cleaned_text'].str.strip() == '')]
        df = df[~(df['cleaned_text'].str.strip() == '')]
        
        df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
        output_path = config['data']['processed_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info('Data preprocessed and save successfully')
        return df
    except Exception as e:
        logging.error(f'Error in data preprocessing: {e}')
        raise

if __name__ == '__main__':
    preprocess_data()
