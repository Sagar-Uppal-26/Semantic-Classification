# 🎬 Semantic Movie Review Classifier (DistilBERT)

An end-to-end Natural Language Processing (NLP) pipeline that classifies complex, highly nuanced movie reviews as Positive or Negative. This project focuses on custom PyTorch training pipelines, tokenization, and Transformer-based feature extraction.

## 🧠 Architecture & Tech Stack
* **Deep Learning Framework:** PyTorch
* **Transformer Model:** `distilbert-base-uncased` (Hugging Face)
* **Data Processing:** Hugging Face `transformers` Tokenizer, Pandas
* **Metrics:** Achieved ~87% accuracy on the highly nuanced IMDB movie review dataset.

## 🛠️ Engineering Highlights
* **Model Selection:** Chose DistilBERT to retain 97% of BERT's original language understanding capabilities while reducing the model parameters by 40% and increasing training/inference speed by 60%. 
* **Custom Tokenization & Data Loaders:** Implemented advanced PyTorch datasets with dynamic padding, attention masks, and truncation strategies (max length 512) to efficiently handle variable-length string inputs in batches.
* **Custom Training Loop:** Engineered a manual PyTorch training loop managing hardware device mapping (CPU/GPU), gradient tracking, Adam optimization, cross-entropy loss calculation, and live accuracy evaluation across epochs.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Semantic-Movie-Review-Classification.git](https://github.com/your-username/Semantic-Movie-Review-Classification.git)
   cd Semantic-Movie-Review-Classification
2. Install the required dependencies:
   ```bash
   pip install torch transformers pandas scikit-learn jupyter
3. Open the Jupyter Notebook to run the training pipeline or test the inference cells:
   ```bash
   jupyter notebook movie_sentiment_distilbert.ipynb
(Note: Ensure you have a GPU environment like Google Colab enabled for faster training).
