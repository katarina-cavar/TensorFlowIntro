
import tensorflow as tf 
print(tf.__version__)
import tensorflow_datasets as tfds 

imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

train_data, test_data = imdb["train"], imdb["test"]

tokenizer = info.features["text"].encoder

print(tokenizer.subwords)

# // online in notebook... downloads and trains too slow