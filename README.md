# 🎬 IMDB Movie Review Sentiment Analysis

This project performs **sentiment analysis** on the IMDB Movie Reviews dataset using **Natural Language Processing (NLP)** and **Logistic Regression**.  
It preprocesses text data, visualizes word distributions with **WordClouds**, trains a **TF-IDF + Logistic Regression** model, and saves the trained model for future use.

---

## 📌 Project Overview

- **Dataset:** IMDB Movie Reviews (50,000 reviews)
- **Task:** Binary Sentiment Classification  
  - Positive → `1`  
  - Negative → `0`
- **Model:** Logistic Regression
- **Feature Extraction:** TF-IDF Vectorizer
- **Visualization:** WordCloud
- **Model Saving:** Pickle

---

## 📂 Dataset

The dataset contains **50,000 movie reviews** with balanced sentiment labels.

| Column    | Description |
|----------|-------------|
| review   | Movie review text |
| sentiment| positive / negative |

File used:
IMDB Dataset.csv

yaml
Copy code

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- NLTK
- Scikit-learn
- WordCloud
- Pickle

---

## 🔄 Project Workflow

### 1️⃣ Data Loading
- Load the dataset using Pandas
- Remove missing values

### 2️⃣ Text Cleaning
- Convert text to lowercase
- Remove English stopwords using **NLTK**

### 3️⃣ Data Visualization
- Generate **WordClouds** for:
  - Positive reviews
  - Negative reviews

### 4️⃣ Feature Extraction
- Convert reviews into numerical features using **TF-IDF**
- Limit features to **2,500 most important words**

### 5️⃣ Model Training
- Split data into training and testing sets (80/20)
- Train a **Logistic Regression** classifier

### 6️⃣ Model Evaluation
- Predict sentiment on test data
- Display **Confusion Matrix**

### 7️⃣ Model Saving
- Save the trained vectorizer and model using **pickle**

---

## 📊 Model Evaluation

- Balanced dataset ensures unbiased learning
- Confusion Matrix used to evaluate predictions
- Suitable baseline model for sentiment analysis

---

## 💾 Saved Model

The trained model is saved as:

model.pkl

yaml
Copy code

This file contains:
- TF-IDF Vectorizer
- Logistic Regression Model

It can be loaded later for inference without retraining.

---

## ▶️ How to Run the Project

### 1. Install Required Libraries
``bash
pip install pandas numpy matplotlib nltk scikit-learn wordcloud

---

### 2. Download NLTK Stopwords
python
Copy code
import nltk
nltk.download('stopwords')

---

### 3. Run the Notebook
Execute all cells in the Jupyter Notebook sequentially.

---

### 🔮 Future Enhancements
Apply lemmatization or stemming

Try advanced models (SVM, Naive Bayes, LSTM)

Deploy the model using Flask or Streamlit

Add accuracy, precision, recall, and F1-score metrics

---

### 📜 License
This project is open-source and intended for learning and educational purposes.

---
### 👤 Author
Rahul Yadav
