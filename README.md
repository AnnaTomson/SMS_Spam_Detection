# SMS Spam Detection System Using NLP
## Overview
This project is an SMS Spam Detection System that uses Natural Language Processing (NLP) techniques and a Naive Bayes Classifier to classify messages as spam or ham (not spam). The system can analyze messages, predict their category, and enhance spam filtering efficiency.

## Features
- Detect whether a given SMS message is Spam or Ham.
- Interactive input for real-time predictions.
- Uses Naive Bayes algorithm and CountVectorizer for text classification.
- High accuracy and adaptability to various types of spam messages.

## Dataset
The project uses a labeled SMS dataset containing two fields:

- class: Indicates whether the message is "ham" (not spam) or "spam".
- message: The text content of the SMS.

## Requirements
To run this project, you need the following Python packages:

- pandas: For data manipulation.
- scikit-learn: For machine learning algorithms.
- numpy: For numerical computations.

## Model Details
- Algorithm: Naive Bayes Classifier.
- Vectorizer: CountVectorizer for converting text into numerical features.
- Training-Test Split: 80% training, 20% testing.

## Future Enhancements
- Use TF-IDF Vectorizer for improved text representation.
- Explore deep learning models like LSTMs for enhanced performance.
- Add multilingual SMS spam detection.
