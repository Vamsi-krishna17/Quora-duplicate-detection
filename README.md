<<<<<<< HEAD
# 🔍 Semantic Similarity Engine — Duplicate Question Detection

A comparative NLP project that benchmarks five different text-representation approaches — from classical Bag-of-Words to fine-tuned Transformers — on the task of detecting whether two differently-worded questions mean the same thing.

**Live demo:** *(add your Streamlit Cloud link here after deploying)*

---

## Problem Statement

Given two questions, predict whether they are **duplicates** (same underlying meaning) or **not**. This is a real-world problem for platforms like Quora, StackOverflow, and customer-support systems, where the same question is often asked in many different ways.

Example:
| Question 1 | Question 2 | Duplicate? |
|---|---|---|
| "What is the main function of protons in the nucleus?" | "What is the main function of a nucleus?" | No |
| "How do I write blue letters in Instagram bio?" | "How do people write in blue big words?" | Yes |

Plain keyword matching fails on both cases — this is why NLP models that understand **meaning**, not just words, are needed.

---

## Project Flow

The project follows the historical evolution of NLP text representation, testing each approach on the exact same dataset and task for a fair comparison.

```
Raw Data (Quora Question Pairs)
        │
        ▼
1. Data Sampling — balanced 50K sample (25K duplicate + 25K non-duplicate)
        │
        ▼
2. Preprocessing — clean text, expand contractions, tokenize, pad sequences,
   stratified train/val/test split
        │
        ▼
3. Classical Baselines (Logistic Regression classifier on top of each):
   ├── Bag-of-Words (word counts)
   ├── TF-IDF (word importance weighting)
   └── Word2Vec (trained on own corpus, averaged sentence vectors)
        │
        ▼
4. Sequence Models (Siamese architecture — shared weights, two input branches):
   ├── Siamese LSTM
   └── Siamese GRU
        │
        ▼
5. Transformer:
   └── Fine-tuned DistilBERT (pretrained, attention-based, context-aware)
        │
        ▼
6. Comparison — accuracy, F1, precision, recall, training time, parameter count
   across all 6 models
        │
        ▼
7. Deployment — Streamlit web app for live interactive testing
```

### Why this progression?
Each stage removes a limitation of the previous one:
- **BoW/TF-IDF** → no understanding of word meaning, only word overlap
- **Word2Vec** → captures word meaning, but loses word order/sequence
- **LSTM/GRU** → captures sequence and word order
- **Transformer (DistilBERT)** → captures full-sentence context via self-attention, understands which words relate to which, regardless of position

---

## Dataset

[Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) — labeled dataset of question pairs from Kaggle. Human-labeled, ~404K pairs originally; a balanced 50K subset (25K duplicate, 25K non-duplicate) was used here for faster iteration across multiple model architectures.

---

## Results

| Model | Accuracy | F1 | Precision | Recall | Train Time (s) | Params |
|---|---|---|---|---|---|---|
| BoW + Logistic Regression | 0.758 | 0.766 | 0.742 | 0.792 | 39.5 | — |
| TF-IDF + Logistic Regression | 0.745 | 0.753 | 0.729 | 0.779 | 6.8 | — |
| Word2Vec (avg) + Logistic Regression | 0.730 | 0.730 | 0.732 | 0.728 | 1.4 | — |
| Siamese LSTM | 0.739 | 0.750 | 0.720 | 0.782 | 30.5 | 2,050,561 |
| Siamese GRU | 0.726 | 0.741 | 0.701 | 0.786 | 30.1 | — |
| Fine-tuned DistilBERT | *(fill in after running)* | | | | | |

*(Full results in `baseline_results.csv`)*

---

## Tech Stack

- **Data processing:** pandas, numpy, scikit-learn
- **Classical NLP:** CountVectorizer, TfidfVectorizer, Gensim (Word2Vec)
- **Deep learning:** TensorFlow / Keras (Siamese LSTM, GRU), HuggingFace Transformers (DistilBERT)
- **Deployment:** Streamlit

---

## Repository Structure

```
├── app.py                     # Streamlit web app (Siamese LSTM demo)
├── requirements.txt
├── tokenizer.pkl               # Keras tokenizer (for LSTM/GRU)
├── siamese_lstm.h5             # trained LSTM model weights
├── baseline_results.csv        # full comparison table across all models
├── notebooks/
│   ├── 01_load_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baselines_bow_tfidf_word2vec.ipynb
│   ├── 04_siamese_lstm.ipynb
│   ├── 05_siamese_gru.ipynb
│   └── 06_distilbert.ipynb
└── README.md
```

---

## How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/quora-duplicate-detection.git
cd quora-duplicate-detection
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Vamsi Krishna**

# Quora-duplicate-detection
858460d8cb67d674977ad917d8a079877c84d976
