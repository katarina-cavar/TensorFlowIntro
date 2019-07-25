"""
In the videos you looked at how you would improve Fashion MNIST using Convolutions. For your exercise see if you can 
improve MNIST to 99.8% accuracy or more using only a single convolutional layer and a single MaxPooling 2D. You should 
stop training once the accuracy goes above this amount. It should happen in less than 20 epochs, so it's ok to hard code 
the number of epochs for training, but your training must end once it hits the above metric. If it doesn't, then you'll 
need to redesign your layers.

I've started the code for you -- you need to finish it!

When 99.8% accuracy has been hit, you should print out the string "Reached 99.8% accuracy so cancelling training!"
"""

import tensorflow as tf 


class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch,logs={}):
		if(logs.get("acc")>0.998):
			print("\nReached 99.8% accuracy so cancelling training")
			self.model.stop_training=True

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train, x_test = x_train/255.0, x_test/255.0
print(x_train.shape)

callbacks = myCallback()

model = tf.keras.models.Sequential([
  	tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  	tf.keras.layers.MaxPooling2D(2, 2),
  	tf.keras.layers.Flatten(),
  	tf.keras.layers.Dense(128, activation='relu'),
  	tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",
	loss="sparse_categorical_crossentropy",
	metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])