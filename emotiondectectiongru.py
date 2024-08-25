# Step 1: Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,GRU
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
nltk.download('wordnet')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 2: Load the CSV file
data = pd.read_pickle("/content/drive/MyDrive/Colab Notebooks/merged_training.pkl")

data.reset_index(drop=True, inplace=True)

# Step 3: Preprocess the data
text = data['text'].tolist()

X=data['text']
labels = data['emotions'].tolist()

#Removeing stop words and doing stemming
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#texts = []
texts = []
for i in range(0, len(X)):
    review = re.sub('[^a-zA-Z]', ' ', X[i])
    #review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review  if not word in stopwords.words('english')]
    #review = [lemmatizer.lemmatize(word) for word in review]
    review = ' '.join(review)
    texts.append(review)

# Step 4: Perform label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

#Split the train set into train and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Step 6: Tokenize the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

vocab_size = len(tokenizer.word_index) + 1

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Step 7: Pad the sequences
max_sequence_length = 100

X_train_padded = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_sequence_length)
X_val_padded = pad_sequences(X_val_seq, maxlen=max_sequence_length)
# Step 8: Load the pre-trained word embeddings (GloVe)
embedding_dim = 100

# Download and place the pre-trained GloVe embeddings file (e.g., glove.6B.100d.txt) in the same directory
embedding_path = '/content/drive/MyDrive/Colab Notebooks/glove.6B.100d.txt'

embeddings_index = {}
with open(embedding_path, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Step 9: Build the classification model using pre-trained embeddings
import tensorflow as tf
start_time = tf.timestamp()
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(GRU(128))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
end_time = tf.timestamp()

# Step 10: Train the model
batch_size = 64
epochs = 10

history = model.fit(X_train_padded, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val_padded, y_val))


# Step 11: Evaluate the model
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

import matplotlib.pyplot as plt

# Assuming you have recorded the loss values during training in the 'history' object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
test_loss = model.evaluate(X_test_padded, y_test)[0]  # Calculate the test loss

# Create a list of epoch numbers
epochs = range(1, len(train_loss) + 1)

# Plot training loss, validation loss, and test loss
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.axhline(y=test_loss, color='g', linestyle='-', label='Test Loss')
plt.title('Training, Validation, and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
test_accuracy = model.evaluate(X_test_padded, y_test)[1]  # Calculate the test accuracy

# Create a list of epoch numbers
epochs = range(1, len(train_accuracy) + 1)

# Plot training accuracy, validation accuracy, and test accuracy
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.axhline(y=test_accuracy, color='g', linestyle='-', label='Test Accuracy')
plt.title('Training, Validation, and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_pred = model.predict(X_test_padded)

# Convert numerical labels back to original class labels
predicted_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
actual_labels = label_encoder.inverse_transform(y_test)

# Create confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

report = classification_report(y_test, np.argmax(y_pred, axis=1))
print("Classification Report:")
print(report)

print(model.summary())