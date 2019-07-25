import tensorflow as tf 
print(tf.__version__)
# tf.enable_eager_execution()

import tensorflow_datasets as tfds 
import numpy as np 



imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb["train"], imdb["test"]

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data: 
	training_sentences.append(str(s.numpy()))
	training_labels.append(l.numpy())

for s,l in test_data:
	testing_sentences.append(str(s.numpy()))
	testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

"""
	Training and testing our code
"""

# Hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRUNC_TYPE = "post"
OOV_TOK = "<OOV>"
NUM_EPOCHS = 10

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = VOCAB_SIZE, oov_token = OOV_TOK)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=MAX_LENGTH,truncating=TRUNC_TYPE)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=MAX_LENGTH)

reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length = MAX_LENGTH), # key to text sentiment analysis in tf
	tf.keras.layers.Flatten(), # or tf.keras.layers.GlobalAveragePooling1D(),-> simpler and a bit faster
	tf.keras.layers.Dense(6, activation = "relu"),
	tf.keras.layers.Dense(1, activation = "sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(padded, training_labels_final, epochs = NUM_EPOCHS, validation_data = (testing_padded, testing_labels_final))

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, VOCAB_SIZE):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

sentence = "I really think this is amazing. honest."
sequence = tokenizer.texts_to_sequences(sentence)
print(sequence)