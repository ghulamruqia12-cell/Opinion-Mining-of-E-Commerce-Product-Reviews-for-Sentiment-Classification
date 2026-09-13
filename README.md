# Opinion Mining of E-Commerce Product Reviews for Sentiment Classification

## Project Overview

This project implements an end-to-end **Opinion Mining and Sentiment Classification system** for e-commerce product reviews using Natural Language Processing (NLP), Machine Learning, and Information Retrieval (IR) techniques.

The system analyzes customer reviews from the Datafiniti Amazon Reviews dataset and classifies customer feedback into three sentiment categories: **Happy, OK, and Unhappy**. It also retrieves and ranks relevant product reviews using TF-IDF and BM25 retrieval models.

## Objectives

- Analyze customer opinions and sentiments from e-commerce product reviews.
- Preprocess and clean textual review data.
- Build machine learning models for sentiment classification.
- Compare TF-IDF and BM25 for relevant review retrieval.
- Evaluate classification and information retrieval performance.

## Key Features

### 1. Text Preprocessing

- Text cleaning and normalization.
- Tokenization.
- Stopword removal.
- Stemming.
- TF-IDF vectorization.

### 2. Sentiment Classification

- SGD Classifier.
- Support Vector Machine (SVM).
- Sentiment labels based on review ratings.
- Evaluation using Accuracy, Precision, Recall, and F1-Score.

### 3. Information Retrieval

- TF-IDF cosine similarity retrieval.
- BM25 (Okapi) ranking.
- Top-k relevant review retrieval.
- Comparison of retrieval performance.

### 4. Data Visualization

- Classification performance visualizations.
- BM25 score plots.
- TF-IDF vs BM25 retrieval comparisons.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- NLTK
- Rank-BM25
- Matplotlib and Seaborn
- Jupyter Notebook

## Dataset

**Datafiniti Amazon Reviews Dataset**

The dataset contains e-commerce product information, ratings, and customer review text.
Link: (https://www.kaggle.com/datasets/datafiniti/grammar-and-online-product-reviews)

## Key Findings

- The SGD Classifier outperformed SVM across the evaluated classification metrics.
- BM25 retrieved more relevant top-k reviews than TF-IDF in the project’s retrieval comparison.

## Project Structure

```text
Opinion-Mining-of-E-Commerce-Product-Reviews-for-Sentiment-Classification/
│
├── data/
├── preprocessing.py
├── model_training.ipynb
├── bm25_retrieval.ipynb
├── visualizations/
├── requirements.txt

This project demonstrates the application of NLP, machine learning, and information retrieval techniques to analyze customer feedback and retrieve relevant product reviews from e-commerce data.
