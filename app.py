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

# ===== Cấu hình logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ===== Global variables =====
lstm_model = None
word2vec_model = None
nlp = None
stop_words = None
models_loaded = False

# ===== Load corpus gốc =====
data = pd.read_csv('cleaned_data.csv') #-> Các câu gốc
source_texts = data['source_txt'].tolist()

# TF-IDF để tìm câu gốc gần nhất -> vì cần 2 câu để so sánh theo ngữ cảnh
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(source_texts)



#Tìm câu gốc gần câu đạo văn
def find_closest_source(text: str) -> str:
    try:
        query_vec = tfidf_vectorizer.transform([text])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix)
        idx = cosine_sim.argmax()
        logger.info(f"similarity {cosine_sim[0][idx]:.4f}")
        return source_texts[idx]
    except Exception as e:
        logger.error(f"Phát hiện lỗi: {str(e)}")
        raise

#Load LSTM và Word2Vec
def load_models():
    global lstm_model, word2vec_model, nlp, stop_words, models_loaded

    try:
        # Load LSTM model
        lstm_model = load_model('best_model.h5')
        logger.info(f"LSTM input shape: {lstm_model.input_shape}")

        # Load Word2Vec model
        with open('word2vec_model.pkl', 'rb') as f:
            word2vec_model = pickle.load(f)
        logger.info(f"Word2Vec vector size: {word2vec_model.vector_size}")

        # NLP tools
        nltk.download('stopwords', quiet=True)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        models_loaded = True
        logger.info("Tải xong model")
    except Exception as e:
        logger.error(f"Phát hiện lỗi: {str(e)}")
        raise


#Load model lần đầu nếu chưa
@app.before_request
def before_first_request():
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
        input_text = request.form['text'].strip()
        if not input_text:
            return render_template('index.html',
                                   error="Please enter some text to check.",
                                   input_text="")

        # tìm câu gốc gần nhất
        closest_source = find_closest_source(input_text)

        # tiền xử lý DL
        tokens_source = preprocess_text(closest_source)
        tokens_input = preprocess_text(input_text)

        # vector embedding
        vec_source = text_to_sequence(tokens_source)
        vec_input = text_to_sequence(tokens_input)

        # Nối 2 vector lại
        combined_vec = np.concatenate((vec_source, vec_input), axis=2)  # (1, timesteps, features*2)
        logger.info(f"Combined vector shape: {combined_vec.shape}")

        # ======= Dự đoán =======
        prediction = lstm_model.predict(combined_vec, verbose=0)[0][0]
        confidence = prediction if prediction > 0.5 else 1 - prediction   # > 0.5 là đạo văn < 0.5 không phải là đạo văn

        result = {
            'text': "Plagiarism Detected" if prediction > 0.5 else "No Plagiarism Detected",
            'confidence': f"{confidence * 100:.1f}%"
        }

        return render_template('index.html',
                               result=result['text'],
                               confidence=result['confidence'],
                               input_text=input_text,
                               closest_source=closest_source)

    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return render_template('index.html',
                               error="An error occurred during processing.",
                               input_text=request.form.get('text', ''))


def preprocess_text(text: str) -> List[str]:
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
    """Chuyển token thành sequence cho LSTM"""
    try:
        max_timesteps = lstm_model.input_shape[1]
        lstm_embedding_dim = lstm_model.input_shape[2] // 2  # Vì ta nối 2 vector
        w2v_embedding_dim = word2vec_model.vector_size

        word_vectors = []
        for word in tokens:
            if word in word2vec_model.wv:
                word_vector = word2vec_model.wv[word]
                # Điều chỉnh vector
                if w2v_embedding_dim != lstm_embedding_dim:
                    if w2v_embedding_dim > lstm_embedding_dim:
                        word_vector = word_vector[:lstm_embedding_dim]
                    else:
                        padded_vector = np.zeros(lstm_embedding_dim)
                        padded_vector[:w2v_embedding_dim] = word_vector
                        word_vector = padded_vector
                word_vectors.append(word_vector)

            if len(word_vectors) >= max_timesteps:
                break

        # Padding nếu cần
        if len(word_vectors) < max_timesteps:
            padding = [np.zeros(lstm_embedding_dim)] * (max_timesteps - len(word_vectors))
            word_vectors.extend(padding)

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
