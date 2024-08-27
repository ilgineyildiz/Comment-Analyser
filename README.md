# Sentiment Analysis on IMDB Reviews
This project performs sentiment analysis on movie reviews from the IMDB dataset using a deep learning model. The goal is to classify the sentiment of each review as either positive or negative.

## Dataset
The dataset used is IMDB Dataset.csv and contains the following columns:
review: Text of the movie review
sentiment: Sentiment of the review (positive or negative)

### Preprocessing
Label Encoding: The sentiment column is encoded as follows:
Positive reviews: 1
Negative reviews: 0

Text Tokenization: Reviews are tokenized using the Tokenizer from Keras. The vocabulary size is limited to 5000 words.

Padding Sequences: Reviews are padded to ensure uniform input length of 200 words.

### Model
A Sequential model is built with the following layers:

Embedding Layer: Converts integer-encoded words into dense vectors of fixed size (128 dimensions).
LSTM Layer: Processes sequences with 128 units, including dropout and recurrent dropout for regularization.
Dense Layer: Outputs a single value with a sigmoid activation function for binary classification.
The model is compiled with the Adam optimizer and binary crossentropy loss function. It is trained for 5 epochs.

### Training
The model is trained on the training set with the following parameters:

Batch Size: 64
Validation Split: 20% of the training data

### Evaluation
The model's performance is evaluated on the test set. The following metrics are reported:

Classification Report: Shows precision, recall, and F1-score for each class.
Confusion Matrix: Displays the number of true positives, true negatives, false positives, and false negatives.

Classification Report:

              precision    recall  f1-score   support

           0       0.82      0.92      0.87      4961
           1       0.91      0.80      0.85      5039

    accuracy                           0.86     10000
    macro avg       0.87      0.86      0.86     10000
    weighted avg       0.87      0.86      0.86     10000
    
Confusion Matrix:
[[4554  407]
 [ 983 4056]]

### Contributing
Feel free to contribute to this project by submitting issues or pull requests.
