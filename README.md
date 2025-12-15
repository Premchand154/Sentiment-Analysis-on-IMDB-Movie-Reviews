# Sentiment Analysis on IMDB Movie Reviews

This repository contains a **sentiment analysis project** using the **IMDB movie reviews dataset**.
Two different deep learning approaches are implemented and compared:

1. **LSTM-based model (TensorFlow / Keras)**
2. **BERT-based transformer model (Hugging Face / PyTorch)**

The goal is to classify movie reviews as **positive** or **negative**.

---

# Project Structure

```
├── Sentiment_Analysis.ipynb   # Main notebook with all experiments
├── models/
│   └── bert_imdb_sentiment/  # Saved fine-tuned BERT model                 
└── README.md                 # Project documentation
```

---

# Dataset

* **IMDB Movie Reviews Dataset**
* 50,000 reviews total

  * 25,000 training samples
  * 25,000 testing samples
* Binary labels:

  * `0` → Negative
  * `1` → Positive

The dataset is loaded using:

* `tensorflow.keras.datasets.imdb` (for the LSTM model)
* Hugging Face tokenizers (for the BERT model)

---

# Models Implemented

# LSTM Model (Keras)

**Steps:**

* Load IMDB dataset (top 10,000 most frequent words)
* Pad sequences to a fixed length
* Build a sequential neural network with:

  * Embedding layer
  * LSTM layer
  * Dropout
  * Dense output layer with sigmoid activation
* Train using binary cross-entropy loss

**Key Parameters:**

* Vocabulary size: `10,000`
* Embedding dimension: `128`
* Max sequence length: `200`
* Epochs: `5`
* Batch size: `64`

---

# BERT Model (Transformers)

**Steps:**

* Use a pretrained BERT model from Hugging Face
* Tokenize text using `BertTokenizer`
* Create a custom PyTorch dataset
* Fine-tune BERT using Hugging Face `Trainer`
* Evaluate model performance during training
* Save the trained model and tokenizer

**Key Components:**

* `BertForSequenceClassification`
* `Trainer` and `TrainingArguments`
* Padding and truncation to max length of `128`

**Model Output:**

```
models/bert_imdb_sentiment/
```

---

# Technologies Used

* **Python**
* **TensorFlow / Keras**
* **PyTorch**
* **Hugging Face Transformers**
* **NumPy**
* **Jupyter Notebook**

---

# How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow torch transformers datasets
   ```

3. Open the notebook:

   ```bash
   jupyter notebook Sentiment_Analysis.ipynb
   ```

4. Run cells sequentially to:

   * Train the LSTM model
   * Fine-tune the BERT model
   * Save trained models

---

# Results

* **LSTM model** provides a strong baseline for sentiment classification.
* **BERT model** achieves higher accuracy by leveraging contextual embeddings and transfer learning.
* The notebook allows easy experimentation and comparison between traditional RNNs and transformer-based models.

---

# Future Improvements

* Add model evaluation metrics (precision, recall, F1-score)
* Include inference script for new reviews
* Add visualizations for training performance
* Deploy model as an API or web app

---
