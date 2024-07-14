# **Movie Review Sentiment Analysis**

This project implements sentiment analysis for movie reviews using machine learning models and natural language processing techniques.

## Overview

The code performs sentiment analysis on movie reviews using several preprocessing steps and machine learning models. It includes:

- Data cleaning
- Tokenization
- Stop words removal
- Lemmatization
- Bigram creation
- TF-IDF vectorization
- Sentiment prediction using machine learning models (SVM, Logistic Regression, Random Forest, Sequential Model)

The best-performing configuration achieved an accuracy of 87% using TF-IDF with bigrams.

## Dependencies

Ensure you have the following libraries installed:

- pandas
- nltk
- numpy
- scikit-learn
- tensorflow
- keras

You can install these dependencies using pip:
pip install pandas nltk numpy scikit-learn tensorflow keras


## Usage

1. **Data Preprocessing:**
   - Run `data_cleaning(text)` to clean the text data.
   - Use `tokenzation(cleaned_text)` to tokenize the cleaned text.
   - Apply `stop_words_removal(tokenz)` to remove stop words.
   - Perform `lemmatization(words)` to lemmatize the words.

2. **Feature Engineering:**
   - Create bigrams using `create_bigrams(lemmas)`.

3. **TF-IDF Vectorization:**
   - Apply TF-IDF using `apply_tfidf(bi_grams)` with pre-trained vectorizer (`bigram_tfidf_vectorizer.joblib`).

4. **Sentiment Prediction:**
   - Predict sentiment using `predict_sentiment(features)` with a pre-trained model (`sequential_model.h5`).

5. **Example Usage:**
   ```python
   from your_module import processing

   # Example text
   text = "This movie was amazing!"

   # Process the text and get sentiment prediction
   result = processing(text)
   print(result)  # Output: "Positive Review"

 
