from flask import Flask, render_template, request
import numpy as np
import re
import contractions
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from keras.models import load_model
import pickle
import logging
from typing import List

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
lstm_model = None
word2vec_model = None
nlp = None
stop_words = None
models_loaded = False

# ===== Load corpus gốc =====
data = pd.read_csv('cleaned_data.csv')
source_texts = data['processed_source_txt'].tolist()

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(source_texts)


def find_closest_source(text: str) -> str:
    """Tìm câu gốc giống nhất với đoạn nghi đạo văn"""
    try:
        query_vec = tfidf_vectorizer.transform([text])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
        idx = cosine_sim.argmax()
        logger.info(f"Closest source found with similarity {cosine_sim[0][idx]:.4f}")
        return source_texts[idx]
    except Exception as e:
        logger.error(f"Error finding closest source: {str(e)}")
        raise


def load_models():
    """Load all required models"""
    global lstm_model, word2vec_model, nlp, stop_words, models_loaded

    try:
        # Load LSTM model
        lstm_model = load_model('best_model.h5')
        logger.info(f"LSTM model input shape: {lstm_model.input_shape}")

        # Load Word2Vec model
        with open('word2vec_model.pkl', 'rb') as f:
            word2vec_model = pickle.load(f)
        logger.info(f"Word2Vec vector size: {word2vec_model.vector_size}")

        # Initialize NLP tools
        nltk.download('stopwords', quiet=True)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        models_loaded = True
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


@app.before_request
def before_first_request():
    """Ensure models are loaded before handling requests"""
    global models_loaded
    if not models_loaded:
        load_models()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if not models_loaded:
        return render_template('index.html',
                               error="Models are not loaded yet. Please try again.",
                               input_text="")

    try:
        text = request.form['text'].strip()
        if not text:
            return render_template('index.html',
                                   error="Please enter some text to check",
                                   input_text="")

        # ======= Tìm câu gốc gần nhất =======
        closest_source = find_closest_source(text)

        # ======= Preprocessing =======
        tokens_source = preprocess_text(closest_source)
        tokens_input = preprocess_text(text)

        # ======= Vector hóa =======
        vec_source = text_to_sequence(tokens_source)
        vec_input = text_to_sequence(tokens_input)

        # ======= Nối 2 vector theo chiều features =======
        combined_vec = np.concatenate((vec_source, vec_input), axis=2)  # (1, timesteps, features*2)
        logger.info(f"Combined vector shape: {combined_vec.shape}")

        # ======= Prediction =======
        prediction = lstm_model.predict(combined_vec, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction

        result = {
            'text': "✅ Plagiarism Detected" if prediction > 0.5 else "✅ No Plagiarism Detected",
            'confidence': f"{confidence * 100:.1f}%"
        }

        return render_template('index.html',
                               result=result['text'],
                               confidence=result['confidence'],
                               input_text=text,
                               closest_source=closest_source)

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return render_template('index.html',
                               error="An error occurred during processing.",
                               input_text=request.form.get('text', ''))


def preprocess_text(text: str) -> List[str]:
    """Text preprocessing pipeline"""
    try:
        text = text.lower()
        text = contractions.fix(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = [word for word in text.split() if word not in stop_words]
        doc = nlp(" ".join(tokens))
        return [token.lemma_ for token in doc if token.lemma_.strip()]
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise


def text_to_sequence(tokens: List[str]) -> np.ndarray:
    """Convert tokens to LSTM input sequence"""
    try:
        max_timesteps = lstm_model.input_shape[1]
        lstm_embedding_dim = lstm_model.input_shape[2] // 2  # Vì ta sẽ nối 2 vector
        w2v_embedding_dim = word2vec_model.vector_size

        word_vectors = []
        for word in tokens:
            if word in word2vec_model.wv:
                word_vector = word2vec_model.wv[word]

                # Điều chỉnh kích thước vector nếu cần
                if w2v_embedding_dim != lstm_embedding_dim:
                    if w2v_embedding_dim > lstm_embedding_dim:
                        word_vector = word_vector[:lstm_embedding_dim]
                    else:
                        adjusted_vector = np.zeros(lstm_embedding_dim)
                        adjusted_vector[:w2v_embedding_dim] = word_vector
                        word_vector = adjusted_vector

                word_vectors.append(word_vector)

            if len(word_vectors) >= max_timesteps:
                break

        if not word_vectors:
            word_vectors = [np.zeros(lstm_embedding_dim)]

        if len(word_vectors) < max_timesteps:
            padding = [np.zeros(lstm_embedding_dim)] * (max_timesteps - len(word_vectors))
            word_vectors.extend(padding)
        else:
            word_vectors = word_vectors[:max_timesteps]

        return np.array(word_vectors).reshape(1, max_timesteps, lstm_embedding_dim)

    except Exception as e:
        logger.error(f"Sequence conversion error: {str(e)}")
        raise


if __name__ == '__main__':
    try:
        load_models()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
