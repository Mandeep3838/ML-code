# -*- coding: utf-8 -*-


# Importing the Keras Packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
    

# Initialising the CNN
classifier = Sequential()


# Step 1 - Convolution 
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3) , activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())


# Step 4 - Full Connention
classifier.add(Dense(units = 128  , activation = "relu" ))
classifier.add(Dense(units = 1 , activation = "sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'training',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
                                        
test_set = test_datagen.flow_from_directory(
                                            'test',
                                            target_size=(64,64),
                                            batch_size = 32,
                                            class_mode='binary')

classifier.fit_generator(
                        training_set,
                        steps_per_epoch=4832,
                        epochs=3,
                        validation_data=test_set,
                        validation_steps=168)




