#! /usr/bin/python3
# Import Required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
import os
import numpy as np
import cv2
import urllib.request
import tarfile
import sys
import getopt
import configparser
import CropDataFrame as cdf
import random
import pandas as pd
from unettiny2_leo import *
from attention_leo import *

def predict(model, image_path):
                  # Read the image
                  # Read the rose image
                  img = cv2.imread(image_path)

                  # Convert image BGR TO RGB, since OpenCV works with BGR and tensorflow in RGB.
                  #imgrgb = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

                  # Normalize the image to be in range 0-1 and then convert to a float array.
                  test_image = np.array([img]).astype('float64')/ 255.0

                  batch_holder = np.zeros((1, img.shape[0], img.shape[1], 3))
                 # test_image = image.load_img(image_path, target_size = (IMG_HEIGHT,IMG_WIDTH))

          #            test_image = test_image.convert("RGB")
                 # test_image = image.img_to_array(test_image) / 255
           #       np.savetxt('arry2.txt',test_image.flatten())
                  #test_image = np.expand_dims(test_image, axis = 0)
                  x = random.randrange(0,test_image.shape[1] - 512 )
                  y = random.randrange(0,test_image.shape[2] - 512)


          #        batch_holder[0, :] = test_image [0,x:x+512,y:y+512,:]

              #    tf.keras.applications.imagenet_utils.preprocess_input(
              #       test_image, data_format=None, mode='tf'
              #    )


                  #predict the result
                  Out = model.predict(batch_holder)
                  index = tf.argmax(input=Out, axis=-1)
                  # Get the top predicted index
                  classification = colors[index]
                  cv2.imwrite(classification,image_path + '.class.png')
                  return classification






# This is the directory path where all the class folders are
dir_path =  '/home/leo/backup/mucche/'
modelname = 'dummy'
IMG_HEIGHT = 280
IMG_WIDTH = 1024
w = None
h = None
nettypes = ["unetsara", "attention"]

# Define the epoch number
epochs = 60

def usage():
      print ('test.py -d <imagedir> -g -o <modelname> -t <train> -v <val> -w width -h height -e epochs -l[oad_net](no_ext) --net nettype')
      print(nettypes)
      sys.exit(2)


if  len(sys.argv) < 5:
   sys.argv.append("-?")

reload = 0
try:
    opts, args = getopt.getopt(sys.argv[1:],"?l:d:o:w:h:e:t:v:n:s:g",["datadir=","ofile=","width","height","epoch","flow","size","net=","lr="])
except getopt.GetoptError as err:
      print(err)
      usage()
      
val_path = None
csv_path = None
classes = 5
size = 16
gabor = None
nettype = 0
lr = -1
for opt, arg in opts:
      print(opt)
      if opt == '-?':
         usage()
      elif opt in ("-d", "--datadir"):
         dir_path = arg
      elif opt in ("-s", "--size"):
         size = int(arg)
      elif opt in ("-t"):
         print(arg)
         csv_path = arg
      elif opt in ("-v"):
         val_path = arg
      elif opt == ("--lr"):
         lr = float(arg)
      elif opt in ("-o", "--ofile"):
         modelname = arg
      elif opt == "-l":
         reload = arg
      elif opt == '-g':
         gabor = "image_g"
      elif opt in ('-w', "--width"):
         w = int(arg)
         IMAGE_WIDTH = w
      elif opt in ('-h', "--height"):
         h = int(arg)
         IMAGE_HEIGHT = h
      elif opt in ('-n'):
         classes = int(arg)
      elif opt in ("--net"):
         nettype = -1
         for i in range(len(nettypes)):
            if (nettypes[i] == arg):
               nettype = i
         if nettype == -1:
            print("unknown net :", arg)
            usage()
      elif opt in ('-e', "--epoch"):
         epochs = int(arg)
      else:
         print(opt)


h5name = modelname # + '.h5'
modelonnx = modelname + '.onnx'
modelinfo = modelname + '.ini'

# Set the batch size, width, height and the percentage of the validation split.
batch_size = 1
split = 0.2

colors = np.array([
                       [   0,   0, 255],
                       [   0, 255,   0],
                       [ 255,   0,   0],
                       [   0, 170, 255],
                       [ 255,  85, 255]])
labels5 = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
labels3 = np.array([[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,1,0]])
labels4 = np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,1,0,0]])
labels2 = np.array([[1,0],[0,1],[0,1],[0,1],[0,1]])
#labels4 = np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
weights4 = [200, 1, 1, 400]
weights5 = [20, 1, 1, 40, 40]
weights3 = [20, 1, 1]
weights2 = [1, 1]

if (classes == 3):
   labels = labels3
   weights = weights3
elif (classes == 4):
   labels = labels4
   weights = weights4
elif (classes == 2):
   labels = labels2
   weights = weights2
else:
   labels = labels5
   weights = weights5
   
#  Setup the ImagedataGenerator for training, pass in any supported augmentation schemes, notice that we're also splitting the data with split argument.
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split= split,
    #rotation_range=2,
    #width_shift_range=0.05,
    #height_shift_range=0.05,
    #shear_range=0.05,
    #zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect')

# Setup the ImagedataGenerator for validation, no augmentation is done, only rescaling is done, notice that we're also splitting the data with split argument.
datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=split)

if reload != 0:

     config = configparser.ConfigParser()
     config.read(reload + '.ini')
     IMG_WIDTH = int(config['General']['width'])
     IMG_HEIGHT = int(config['General']['height'] )
     oldh5name = reload #+ '.h5'
 
     
f = open(modelinfo,"w")

f.write("[General]\n")
f.write("height = %d\n" % IMG_HEIGHT)
f.write("width = %d\n" % IMG_WIDTH)
f.write("model = %s\n" % modelonnx)
f.close()
print(csv_path)

dataframe = pd.read_csv(csv_path)
if val_path is None:
    # Creating a dataframe with 80%
    # values of original dataframe
    train_set = dataframe.sample(frac = 0.8)

    val_set = dataframe.drop(train_set.index)
else:
    train_set = dataframe
    val_set = pd.read_csv(val_path)


# disable eager execution
#tf.compat.v1.disable_eager_execution()

if reload != 0:
     print('Loading previous model '+ oldh5name)
     model = load_model(oldh5name)
else:
# Here we are creating Sequential model also defing its layers
 # model = transfer_learning(IMG_HEIGHT, IMG_WIDTH , classes)
#  model = seconda_rete(IMG_HEIGHT, IMG_WIDTH , classes)
#  model = terza_rete(IMG_HEIGHT, IMG_WIDTH, classes)
  inputs = 3;
  if gabor is not None:
      inputs = 4
  if (nettypes[nettype] == "unetsara"):
      model = unetsara(input_size = (None, None, inputs), num_classes = classes, startSize = size)
  if (nettypes[nettype] == "attention"):
      model = attention_unet((None, None, inputs), classes)

  #model = unet(input_size = (None, None, inputs), num_classes = classes, startSize = size) 

  #model = rete_unet(inputs, classes, None, None, startSize = size)
  # Compile the model
  #model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
  #model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
  #model.compile(optimizer=Adam(learning_rate=0.001), loss='multinomial_crossentropy', metrics=['accuracy'])

# Data Generation for Training with a constant seed valued 40, notice that we are specifying the subset as 'training'
train_data_generator = cdf.CropDataFrame(model =  model, classes = classes, dataframe = train_set,
                                            image_data_generator=datagen_train,
                                            batch_size=batch_size,
                                            x_col="image",
                                            g_col=gabor,
                                            y_col='mask',
                                            directory=dir_path,
                                            shuffle=True,
                                            seed = 140,
                                            colordict = colors,
                                            labeldict = labels,
                                         #   subset= 'training',
                                            interpolation = 'bicubic',
                                            class_mode = 'raw',
                                            #class_mode = 'categorical',
               #                                          classes = classes,
                                            target_size=(h,w))

# Data Generator for validation images with the same seed to make sure there is no data overlap, notice that we are specifying the subset as 'validation'
vald_data_generator = cdf.CropDataFrame(model = model, classes = classes, dataframe = val_set,
                                        image_data_generator=datagen_val,
                                        batch_size=batch_size,
                                        x_col="image",
                                        g_col=gabor,
                                        y_col='mask',
                                        directory=dir_path,
                                        shuffle=True,
                                        seed = 140,
                                        colordict = colors,
                                        labeldict = labels,
        #                               subset = 'validation',
                                        interpolation = 'bicubic',
                                        class_mode = 'raw',
                                        #class_mode = 'categorical',
                                     #                      classes = classes,
                                        target_size=(h,w))
train_data_generator.next()
vald_data_generator.next()

#for _ in range(len(training_dataset.filenames)):
for z in range(10):
     image, label = train_data_generator.next()

     print(image.shape, label.shape)
     cv2.imwrite("im_" + str(z) + ".jpg", image[0,:,:,:])
     # display the image from the iterator
     for i in range(label.shape[3]):
        cv2.imwrite("im_" + str(z) + "_l_" + str(i) + ".jpg", label[0,:,:,i] * 255)
                  
     
vald_data_generator.reset()
                                
print("Count images: vald")
#count_images(vald_data_generator)
print("Count images: done")
#count_images(train_data_generator)
train_data_generator.reset()
vald_data_generator.reset()

if (lr > 0):
    K.set_value(model.optimizer.learning_rate, 0.001)



#model.summary()
#trainer = unet.Trainer(checkpoint_callback=False)
#trainer.fit(model,
#            train_data_generator,
#            vald_data_generator,
#            epochs=5,
#            batch_size=10)

# Start Training
history = model.fit( train_data_generator,  
      steps_per_epoch= train_data_generator.samples // batch_size, 
      epochs=epochs, validation_data= vald_data_generator,
      #class_weight = weights, 
      validation_steps = vald_data_generator.samples // batch_size 
)

f = open(modelinfo,"w")

f.write("[General]\n")
f.write("height = %d\n" % IMG_HEIGHT)
f.write("width = %d\n" % IMG_WIDTH)
f.write("model = %s\n" % modelonnx)
f.close()


# Use model.fit_generator() if using TF version &lt; 2.2
# Saving your model to disk allows you to use it later


model.save(h5name)

vald_data_generator.reset()

for z in range(10):
     image, label = vald_data_generator.next()
     out = model.predict(image)
     print(image.shape, label.shape)
     cv2.imwrite(modelname + str(z) + ".jpg", image[0,:,:,:])
     # display the image from the iterator
     for i in range(label.shape[3]):
        cv2.imwrite(modelname + str(z) + "_l_" + str(i) + ".jpg", label[0,:,:,i] * 255)
        cv2.imwrite(modelname + str(z) + "_o_" + str(i) + ".jpg", out[0,:,:,i] * 255)
          
#print("Training")
#test_images(model,train_data_generator)

print("Validation")
#test_images(model,vald_data_generator)

sys.exit(0)

# Import keras2onnx and onnx
import onnx
import keras2onnx

# Convert it into onnx
onnx_model = keras2onnx.convert_keras(model, model.name)

# Save the model as onnx
onnx.save_model(onnx_model, modelonnx)

# Plot the accuracy and loss curves for both training and validation

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

def predict_images(data_generator, no = 5):
    sample_training_images, labels = next(data_generator)
 
    # By default we're displaying 15 images, you can show more examples
    total_samples = sample_training_images[:no]
    Out = model.predict(total_samples)

