
# Hate Speech Detection

This repository contains a Jupyter Notebook for detecting hate speech in text data using various machine learning techniques.

## Table of Contents

- [Introduction](#Introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The purpose of this project is to build a machine learning model to classify text data into three categories: "Hate Speech", "Offensive", and "Normal". This is achieved using techniques such as text preprocessing, feature extraction, and training various classifiers.

## Dataset

The project uses multiple datasets for training and evaluation:
- `labeled_data.csv`: Contains tweets labeled as hate speech, offensive, or normal.
- `data2.csv`
- `en_dataset_with_stop_words.csv`

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- nltk
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy nltk scikit-learn
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hate-speech-detection.git
   cd hate-speech-detection
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook hateSpeechDetection.ipynb
   ```

3. Follow the steps in the notebook to preprocess the data, train the models, and evaluate their performance.

### Example Notebook Content

The notebook includes the following steps:

1. **Import Libraries**
   ```python
   import pandas as pd
   import numpy as np
   import nltk
   import re
   import string
   from sklearn.feature_extraction.text import CountVectorizer
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import MultinomialNB
   from sklearn.tree import DecisionTreeClassifier
   from nltk.corpus import stopwords as nltk_stopwords
   ```

2. **Load the Datasets**
   ```python
   data = pd.read_csv("labeled_data.csv")
   data['labels'] = data['class'].map({0: "hate speech", 1: "Offensive", 2: "Normal"})
   data = data[['tweet', 'labels']]

   data2 = pd.read_csv('data2.csv')
   data3 = pd.read_csv('en_dataset_with_stop_words.csv')
   ```

3. **Preprocess and Concatenate Datasets**
   ```python
   concatenated_tweets = pd.concat([data['tweet'], data3['tweet']], axis=0).reset_index(drop=True)
   concatenated_labels = pd.concat([data['labels'], data3['sentiment']], axis=0).reset_index(drop=True)

   # Example of a preprocessing function
   def preprocess_text(text):
       text = text.lower()
       text = re.sub(r'\d+', '', text)
       text = text.translate(str.maketrans('', '', string.punctuation))
       text = text.strip()
       tokens = text.split()
       tokens = [word for word in tokens if word not in nltk_stopwords.words('english')]
       return ' '.join(tokens)

   processed_tweets = concatenated_tweets.apply(preprocess_text)
   ```

4. **Feature Extraction and Model Training**
   ```python
   vectorizer = CountVectorizer()
   X = vectorizer.fit_transform(processed_tweets)

   X_train, X_test, y_train, y_test = train_test_split(X, concatenated_labels, test_size=0.2, random_state=42)

   nb_classifier = MultinomialNB()
   nb_classifier.fit(X_train, y_train)

   dt_classifier = DecisionTreeClassifier()
   dt_classifier.fit(X_train, y_train)
   ```

5. **Model Evaluation**
   ```python
   nb_accuracy = nb_classifier.score(X_test, y_test)
   dt_accuracy = dt_classifier.score(X_test, y_test)

   print(f"Multinomial Naive Bayes Accuracy: {nb_accuracy}")
   print(f"Decision Tree Classifier Accuracy: {dt_accuracy}")
   ```

## Models Used

The following machine learning models are used in this project:

- Multinomial Naive Bayes
- Decision Tree Classifier

## Results

The notebook provides detailed results and visualizations of the model performance, including accuracy, precision, recall, and F1 score.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please contact [laxmikhilnani04@gmail.com](mailto:laxmikhilnani04@gmail.com).
