
# !wget --no-check-certificate \
# 	https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
# 	-O /tmp/sarcasm.json

import json

with open("/tmp/sarcasm.json", "r") as f:
	datastore = json.load(f)

sentences = []
labels = []
# urls = []

for item in datastore:
	sentences.append(item["headline"])
	labels.append(item["is_sarcastic"])
#	urls.append(item["article_link"])

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 32
TRUNC_TYPE = "post"
PADDING_TYPE = "post"
OOV_TOK = "<OOV>"
TRAINING_SIZE = 20000
NUM_EPOCHS = 30

training_sentences = sentences[:TRAINING_SIZE]
testing_sentences = sentences[TRAINING_SIZE:]
training_labels = labels[:TRAINING_SIZE]
testing_labels = labels[TRAINING_SIZE:]

tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token = OOV_TOK)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen = MAX_LENGTH, truncating = TRUNC_TYPE, padding = PADDING_TYPE)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = MAX_LENGTH, truncating = TRUNC_TYPE, padding = PADDING_TYPE)

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(24, activation = "relu"),
	tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.summary()

history = model.fit(training_padded, training_labels, epochs = NUM_EPOCHS, 
	validation_data = (testing_padded, testing_labels))

"""
	Plot the data
"""

import matplotlib.pyplot as plt 

def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history["val_" + string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string, "val_" + string])
	plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

