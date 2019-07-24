"""
Below is code with a link to a happy or sad dataset which contains 80 images, 40 happy and 40 sad. 
Create a convolutional neural network that trains to 100% accuracy on these images,  which cancels 
training upon hitting training accuracy of >.999

Hint -- it will work best with 3 convolutional layers.
"""
import tensorflow as tf 
import os
import zipfile

DESIRED_ACCURACY = 0.999

!wget --no-check-certificate \
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
    -O "/tmp/happy-or-sad.zip"

zip_ref = zipfile.ZipFile("/tmp/happy-or-sad.zip", "r")
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch,logs={}):
		if(logs.get("acc")>= DESIRED_ACCURACY):
			print("\nReached {}% accuracy so cancelling training".format(DESIRED_ACCURACY))
			self.model.stop_training=True


callbacks = myCallback()

# Defining and Compiling the Model
model = tf.keras.models.Sequential([
	# add code
])

from tensorflow.keras.optimizers import RMSprop

model.compile("...add code ... ")

# This code block should create an instance of an ImageDataGenerator called train_datagen 
# And a train_generator by calling train_datagen.flow_from_directory

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = # Your Code Here

train_generator = train_datagen.flow_from_directory(
        # Your Code Here)

# Expected output: 'Found 80 images belonging to 2 classes'


# This code block should call model.fit_generator and train for
# a number of epochs. 
history = model.fit_generator(
      # Your Code Here)
    
# Expected output: "Reached 99.9% accuracy so cancelling training!""