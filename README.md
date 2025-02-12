# SMS Guard: Block Spam, Keep the Important!

![Screenshot 2025-02-12 092441](https://github.com/user-attachments/assets/876bb864-24bc-4021-9f57-9ba33bf7b373)

## Working Link - https://sms-guard.streamlit.app/

## Project Overview

SMS Guard is a machine learning project aimed at identifying and blocking spam messages while keeping important messages. The project uses Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier to classify SMS messages as spam or not spam.

## Project Structure

- `app.py`: Streamlit application for user interaction and message classification.
- `train_model.py`: Script for training the model using the provided dataset.
- `spam.csv`: Dataset containing labeled SMS messages (spam and ham).
- `vectorize2.pkl`: Saved TF-IDF vectorizer.
- `model2.pkl`: Saved Multinomial Naive Bayes model.
- `requirements.txt`: List of dependencies required for the project.
- `sms-spam-detection.ipynb`: Jupyter notebook used for experimenting with different algorithms and preprocessing techniques.

## Steps Performed

1. **Data Loading and Preprocessing**:
   - Loaded the dataset from `spam.csv`.
   - Preprocessed the dataset by selecting relevant columns and mapping labels to numerical values.

2. **Text Transformation**:
   - Used `TfidfVectorizer` to transform the text data into TF-IDF features.

3. **Model Training**:
   - Split the data into training and testing sets.
   - Trained a Multinomial Naive Bayes model on the training data.
   - Evaluated the model using accuracy, precision, recall, and F1 score.

4. **Model Selection**:
   - Experimented with several algorithms including SVM, Random Forest, and Logistic Regression in `sms-spam-detection.ipynb`.
   - Finalized the Multinomial Naive Bayes model based on its performance and simplicity.

5. **Model Saving**:
   - Saved the trained TF-IDF vectorizer and model to `vectorize2.pkl` and `model2.pkl` respectively.

6. **Streamlit Application**:
   - Created a Streamlit application (`app.py`) to allow users to input SMS messages and classify them as spam or not spam.

## Outputs

- The trained model achieved the following evaluation metrics:
  - Accuracy: `0.9806576402321083%`
  - Precision: ` 0.9538461538461539%`

- The Streamlit application provides a user-friendly interface for classifying SMS messages.

## Improvements

- **Data Augmentation**: Increase the size of the dataset by adding more labeled SMS messages to improve model performance.
- **Hyperparameter Tuning**: Experiment with different hyperparameters for the Multinomial Naive Bayes model to improve accuracy.
- **Model Comparison**: Compare the performance of different classifiers (e.g., SVM, Random Forest) to find the best model for this task.
- **Feature Engineering**: Explore additional text preprocessing and feature engineering techniques to improve model performance.

## Evaluation

The model was evaluated using the following metrics:

- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Measures the proportion of true positive predictions among all positive predictions.

## How to Run

1. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
