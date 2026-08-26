import streamlit as st
import pickle
import re
import contractions
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 30
EMBED_DIM = 100
LSTM_UNITS = 64

# ===== REBUILD EXACT SAME ARCHITECTURE AS TRAINING =====
def build_siamese_lstm(vocab_size):
    input_q1 = Input(shape=(MAX_LEN,), name="q1_input")
    input_q2 = Input(shape=(MAX_LEN,), name="q2_input")

    embedding_layer = Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_LEN)
    lstm_layer = LSTM(LSTM_UNITS)

    q1_embed = embedding_layer(input_q1)
    q2_embed = embedding_layer(input_q2)

    q1_encoded = lstm_layer(q1_embed)
    q2_encoded = lstm_layer(q2_embed)

    # output_shape explicitly given now - required in Keras 3
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(LSTM_UNITS,))([q1_encoded, q2_encoded])
    mult = Lambda(lambda x: x[0] * x[1], output_shape=(LSTM_UNITS,))([q1_encoded, q2_encoded])
    merged = Concatenate()([diff, mult])

    dense1 = Dense(64, activation="relu")(merged)
    output = Dense(1, activation="sigmoid")(dense1)

    model = Model(inputs=[input_q1, input_q2], outputs=output)
    return model

# ===== LOAD ARTIFACTS =====
@st.cache_resource
def load_artifacts():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    vocab_size = min(20000, len(tokenizer.word_index) + 1)
    model = build_siamese_lstm(vocab_size)
    model.load_weights("siamese_lstm.h5")  # load only weights into rebuilt architecture
    return tokenizer, model

tokenizer, model = load_artifacts()

# ===== SAME CLEANING FUNCTION AS TRAINING =====
def clean_text(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = re.sub(r"[^a-z0-9\s?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_duplicate(q1, q2):
    q1_clean = clean_text(q1)
    q2_clean = clean_text(q2)

    q1_seq = tokenizer.texts_to_sequences([q1_clean])
    q2_seq = tokenizer.texts_to_sequences([q2_clean])

    q1_pad = pad_sequences(q1_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    q2_pad = pad_sequences(q2_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    prob = model.predict([q1_pad, q2_pad])[0][0]
    label = "Duplicate" if prob > 0.5 else "Not Duplicate"
    return label, prob

# ===== UI =====
st.set_page_config(page_title="Semantic Similarity Engine", page_icon="🔍")

st.title("🔍 Semantic Similarity Engine")
st.markdown("Detect whether two questions mean the same thing — powered by a Siamese LSTM network trained on Quora Question Pairs.")

col1, col2 = st.columns(2)
with col1:
    question1 = st.text_area("Question 1", placeholder="e.g. How can I learn Python?")
with col2:
    question2 = st.text_area("Question 2", placeholder="e.g. What is the best way to learn Python?")

if st.button("Check Similarity", type="primary"):
    if question1.strip() == "" or question2.strip() == "":
        st.warning("Please enter both questions.")
    else:
        label, prob = predict_duplicate(question1, question2)
        confidence = prob if label == "Duplicate" else 1 - prob

        if label == "Duplicate":
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label}")

        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.progress(float(confidence))

st.markdown("---")
st.caption("Model: Siamese LSTM | Dataset: Quora Question Pairs (50K balanced sample) | Also benchmarked: BoW, TF-IDF, Word2Vec, GRU, DistilBERT")