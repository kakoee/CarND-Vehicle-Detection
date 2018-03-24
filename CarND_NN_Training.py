# written by Mohammad Reza Kakoee 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from all_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
import pickle
#from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Activation,Dropout
from keras.layers.convolutional import Convolution2D,MaxPooling2D,Cropping2D
from keras.layers import Conv2D

def extract_images(image_file_list):
    images = []
    for line in image_file_list:

        # adding center image and steering meas
        img_filename=line
        image = cv2.imread(img_filename)
        #print(line[0],line[1])

        #converting image to RGB as cv2 return BGR while drive.py uses RGB    
        imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        images.append(imgRGB)
    X_train = np.array(images)
    return X_train
    

#generator function 
def generator(samplesX, samplesY, batch_size=32):
    num_samples = samplesY.shape[0]

    samples_m= np.column_stack((samplesX,samplesY))

    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples_m)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            targets = []
            for line in batch_samples:

                # adding center image and steering meas
                img_filename=line[0]
                image = cv2.imread(img_filename)
                #print(line[0],line[1])

                #converting image to RGB as cv2 return BGR    
                imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                images.append(imgRGB)
                target = line[1]
                targets.append(target)    
    
                #adding flip version of the image to dataset
                image_flipped = np.fliplr(imgRGB)
                target_flipped = target
                images.append(image_flipped)
                targets.append(target_flipped)

            X_train = np.array(images)
            y_train = np.array(targets)
            yield sklearn.utils.shuffle(X_train, y_train)



## debug
debug_train=0
debug_save_model =1
debug_read_video=0
debug_custom_train_extract=0
BATCH_SIZE=32
EPOCH=8
model_filename = 'finalized_model_NN_rgb.sav'

##



#1st Training

# Read in cars and notcars
images_notcars = glob.glob('./training/non-vehicles/**/*.png')
images_cars_far = glob.glob('./training/vehicles/GTI_Far/*.png')
images_cars_Left = glob.glob('./training/vehicles/GTI_Left/*.png')
images_cars_Right = glob.glob('./training/vehicles/GTI_Right/*.png')
images_cars_MiddleClose = glob.glob('./training/vehicles/GTI_MiddleClose/*.png')
images_cars_KITTI = glob.glob('./training/vehicles/KITTI_extracted/*.png')



cars = []
notcars = []
for image in images_notcars:
    notcars.append(image)
    
for image in images_cars_far:
    cars.append(image)
for image in images_cars_Left:
    cars.append(image)
for image in images_cars_Right:
    cars.append(image)
for image in images_cars_MiddleClose:
    cars.append(image)
for image in images_cars_KITTI:
    cars.append(image)

np.random.shuffle(cars) 
np.random.shuffle(notcars)

cars=cars[0:(int)(len(cars)/2)]
 
#custom_train_car
images_cars_custom = glob.glob('./test_images/Train*')
for image in images_cars_custom:
    cars.append(image)

#custom_train_nocar extraction
if(debug_custom_train_extract==1):
    images_notcars_custom = glob.glob('./test_images_nocar/*.jpg')
    cnt=0
    for imagename in images_notcars_custom:
        not_cars_c=crop_images(imagename,64)
        cnt+=1
        for index,image in enumerate(not_cars_c):
            filename="custom_train_set/custom_nocar"+str(cnt)+"_"+str(index)+".png"
            cv2.imwrite(filename,image)

#custom_train_notcar
images_notcars_custom = glob.glob('./custom_train_set/custom_nocar*')
for image in images_notcars_custom:
    notcars.append(image)

  



# Create an array stack of feature vectors
X = np.hstack((cars, notcars))

# Define the labels vector
y = np.hstack((np.ones(len(cars),dtype=int), np.zeros(len(notcars),dtype=int)))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)


if(debug_train==1):
    
    
    # compile and train the model using the generator function
    train_generator = generator(X_train,y_train, batch_size=BATCH_SIZE)
    validation_generator = generator(X_test,y_test, batch_size=BATCH_SIZE)


    # Build Convolutional Neural Network in Keras
    model = Sequential()
    
    #preprocessing using Lambda layer - normalize and mean
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(64, 64,3)))
    
    
    #first Conv layer with relu and maxpool
    model.add(Convolution2D(8, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    #second Conv layer with relu and maxpool
    model.add(Convolution2D(16, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    #Third Conv layer with relu
    model.add(Convolution2D(24, 3, 3))
    model.add(Activation('relu'))


    
    #first fully connected layer follow up by dropout layer as FC layers tend to overfit
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #second fully connected layer 
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #third fully connected layer 
    model.add(Dense(10))
    #last layer to output
    model.add(Dense(1))    
   
    model.compile(loss='mse', optimizer='adam',metrics=['mse', 'accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=len(X_train)/BATCH_SIZE, validation_data=validation_generator,validation_steps=len(X_test)/BATCH_SIZE, nb_epoch=EPOCH)   

    if(debug_save_model==1):
        model.save(model_filename)    
 
    X_images= extract_images(X_train)
   
    metrics = model.evaluate(X_images, y_train)
    print('')
    print(np.ravel(model.predict(X_images)))
    print(model.metrics_names)
    print(metrics)
    #for i in range(len(model.metrics_names)):
    #    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


else:
    print("==== loading NN model...")
    from keras.models import load_model
    model = load_model(model_filename)
    print("==== load done")

    print("==== Testing NN model...")

    X_images= extract_images(X_test)
  
    metrics = model.evaluate(X_images, y_test)

    print(model.metrics_names)
    print(metrics)
    
    print("one prediction:",model.predict(X_images[1:2]),"---label:",y_test[0])
    
    print("==== Test done")