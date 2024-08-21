Bu projenin amacı IMDB filmlerinin altına yazılan yorumları LSTM kullanarak pozitif ve negatif olarak ikiye ayırmaktır. Veri seti dengeli ve Kaggle'dan alınmıştır.

# Kod Hakkında Açıklamalar

```from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()            
dataset["sentiment"] = le.fit_transform(dataset["sentiment"])
print(dataset)
```
Bu kısımda sentiment kısmını yani output'u pozitif-negatif şeklinden(string/object) 0 ve 1 olarak değiştiriyoruz.

```train_data, test_data = train_test_split(dataset, test_size = 0.2, random_state=42) 
tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_data["review"])
```

Öncelikle train_test_split uyguluyoruz. Ardından tokenizer ile en çok kullanılan 500 kelimeyi alıyoruz.

```X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)
```

Pad_sequences ile metinlerin uzunluğunu standart hale getiriyoruz.maxlen=200 tüm dizileri 200 kelime olacak şekilde pad eder.(kısa dizilere 0 ekeler,uzunları siler.)Derin öğrenme modelleri sabit uzunluktaki girişler ile çalışır.
veri kaybı olmaması için yapılmalıdır.

```Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]
model = Sequential()
model.add(Embedding(input_dim =5000, output_dim = 128, input_length = 200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = "sigmoid"))
```

input_dim ve output_dim embedding katmanının parametreleridir. Tokenizer ile metni ayrıştırınca her kelimeye farklı bir index atanır, input_dim bu indekslerin alabileceği max. değerdir.output_dim her kelimenin kaç boyutlu bir vektörle temsil edileceğini belirler.Recurrent drop_out geri besleme nöronu, overfitting'i engellemek için kullanılır. Sigmod 0-1 aralığında sınırlandırır(çıkış fonksiyonunu) o yüzden onu kullandık.(çıktılarımız 0 veya 1 olduğu için)

```model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

binary_crossentropy ikili sınıflandırma sık kullanılır. O yüzden parametre olarak onu seçtik.
```

# Sonuçlar

Classification Report:

                    precision    recall  f1-score   support

               0       0.82      0.92      0.87      4961
               1       0.91      0.80      0.85      5039

    accuracy                               0.86     10000
    macro avg          0.87      0.86      0.86     10000
    weighted avg       0.87      0.86      0.86     10000

5 epoch, batch_size=64, val_split=0.2 parametreleri ile 0.86 accuracy-f1 sonucuna ulaştık. Optuna kullanılarak yapılacak bir hiperparametre optimizasyonu ile bu sonuç arttırılabilir. 







