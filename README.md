### This project aims to analyse the comments as 'positive' or 'negative' written for IMDB movies using LSTM.
### This dataset is taken from Kaggle.

### Imports:
Before you start you should install these requirements in your environment:
- pandas
- numpy
- scikit-learn(sklearn)
- tensorflow

## Explainig
### Preprocessing
-Label Encoder

### Parameters used
train_data, test_data = train_test_split(dataset, test_size = 0.2, random_state=42) <br/>
model.add(Embedding(input_dim =5000, output_dim = 128, input_length = 200)) <br/>
model.add(LSTM(128, dropout=0.2, recurrent_dropout = 0.2)) <br/>
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"]) <br/>
history=model.fit(X_train, Y_train, epochs = 5, batch_size = 64, validation_split = 0.2) <br/>
p.s You can always get better results with a hiperparameter optimization(optuna would be a good option for NLP projects)

# Sonuçlar

Classification Report:

                    precision    recall  f1-score   support

               0       0.82      0.92      0.87      4961
               1       0.91      0.80      0.85      5039

    accuracy                               0.86     10000
    macro avg          0.87      0.86      0.86     10000
    weighted avg       0.87      0.86      0.86     10000









