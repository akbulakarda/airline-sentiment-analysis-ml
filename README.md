# Sentiment Analysis of Airline Tweets Using Machine Learning

## Overview
This project performs sentiment analysis on airline-related tweets, classifying them into **negative**, **neutral**, and **positive** sentiments using machine learning techniques. We applied various models, including **Logistic Regression**, **Naive Bayes**, **SVM**, and **Random Forest**, with **Random Oversampling (ROS)** and **SMOTE** to handle class imbalance.

## Dataset
The dataset used in this project is the [Airline Tweets Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) available on Kaggle. It contains around 14,000 tweets related to various airlines.

## Key Steps:
1. **Data Cleaning and Preprocessing**: Tokenization, stopword removal, and TF-IDF transformation.
2. **Modeling**: Training and evaluation of multiple models, including:
   - Logistic Regression
   - Naive Bayes
   - Support Vector Machine (SVM)
   - Random Forest (chosen as the final model)
3. **Model Comparison**: Evaluation of accuracy, F1-score, and cross-validation.
4. **Visualization**: Confusion matrix, feature importance, precision-recall curves, and word frequencies for each sentiment.

## Results:
- The **Random Forest (ROS)** model was selected as the best-performing model with an accuracy of **92.7%** and a strong macro F1-score of **0.93**.
- Cross-validation confirmed the model's generalization capability with a mean accuracy of **92.73%**.

## Conclusion:
This project demonstrates how machine learning models can effectively classify sentiments from social media data, offering insights into customer feedback for airline services.

## Instructions to Run:
1. Clone this repository.
2. Run the provided Jupyter Notebook (`airline_sentiment_analysis_machine_learning.ipynb`).
3. Ensure you have all necessary dependencies installed (see the notebook for detailed requirements).
