
#Convolutional Neural Network

#Importing the libraries
#Sequential Package used to Initialize Neural Network
from keras.models import Sequential
#To making first step of CNN - Convolution Layer and images are 2D
from keras.layers import Convolution2D
#Used for MaxPooling step of CNN
from keras.layers import MaxPooling2D
#Used in flattening step of CNN
from keras.layers import Flatten
#Used in fully connected layer of CNN
from keras.layers import Dense

#Initializing the CNN
classifier = Sequential()

#Step-1 -  Adding the Convolutional Layer composed of Feature Maps
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

#Step-2 - Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Adding second Convolutional Layer to increase accuracy and decrease the difference between trainig set and test set accuracy
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#Step-3 - Flattening
classifier.add(Flatten())

#Step-4 - Add fully connected layers (Hidden layers)
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#Adding output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image pre-processing step where we fit CNN to all the images
#Use keras documentation - for image augmentation
from keras.preprocessing.image import ImageDataGenerator
  
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=(8000/32),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=(2000/32))

#Making new predictions
import numpy as np
#from keras.preprocessing import image
#test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = classifier.predict(test_image)
##To get the mapping
#training_set.class_indices
#
#test_image2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
#test_image2 = image.img_to_array(test_image2)
#test_image2 = np.expand_dims(test_image2, axis = 0)
#result2 = classifier.predict(test_image2)

from skimage.io import imread
from skimage.transform import resize
 
class_labels = {v: k for k, v in training_set.class_indices.items()}
 
img = imread('dataset/single_prediction/cat_or_dog_2.jpg') #make sure that path_to_file contains the path to the image you want to predict on. 
img = resize(img,(64,64))
img = np.expand_dims(img,axis=0)
 
img = img/(255.0)
prediction = classifier.predict_classes(img)
 
result = classifier.predict(img)

#Apply evaluation Techniques from ANN