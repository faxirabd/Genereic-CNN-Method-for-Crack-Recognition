import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.metrics import classification_report
import math
#import numpy

# dimensions of our images.
img_width, img_height = 224, 224

#path to the dataset folder with five-fold cross validation folder strucure
path="..\data\\Glasscracks\\D4K\\FiveFold\\"
print("Glass:50 epoch ResNet50")

nb_train_samples = 360
nb_validation_samples = 88
epochs = 50
batch_size = 8
accuracy=[]

for cross in range(1):
  # build the VGG16 network
  datagen = ImageDataGenerator()#rescale=1. / 255) 
  model = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
  # model.summary()
  train_data_dir = path + 'Fold'+str(cross)+'/train/'
  validation_data_dir = path + 'Fold'+str(cross)+'/validation/'
  
  train_generator = datagen.flow_from_directory(
       train_data_dir,
       target_size=(img_width, img_height),
       batch_size=batch_size,
       class_mode=None,
       shuffle=False)
  
  
  bottleneck_features_train = model.predict_generator(
       train_generator, nb_train_samples // batch_size)

  val_generator = datagen.flow_from_directory(
      validation_data_dir,
      target_size=(img_width, img_height),
      batch_size=batch_size,
      class_mode=None,
      shuffle=False)
  
  bottleneck_features_validation = model.predict_generator(
      val_generator, nb_validation_samples // batch_size)
  
  train_data = bottleneck_features_train
  train_labels = np.array(
       [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
  
  # seed = 7
  #X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.2,
  #                                                     random_state=seed)
  
  validation_data = bottleneck_features_validation
  validation_labels = np.array(
      [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

  model = Sequential()
  model.add(Flatten(input_shape=validation_data.shape[1:]))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.summary()
  model_json = model.to_json()
  with open("FullyConnectedTranfer.json", "w") as json_file:
    json_file.write(model_json)
 
  
  model.compile(optimizer='rmsprop',
                loss='binary_crossentropy', metrics=['accuracy'])
  
  history=model.fit(train_data, train_labels,
             epochs=epochs,
             batch_size=batch_size)#, verbose=0)
                #validation_data=(validation_data, validation_labels))
  
  #model.save_weights("DHD\\VGG16Transfer\\"+str(cross)+".h5")
  #model.load_weights("All\\ResNetDHD"+str(cross)+".h5")
  #model.summary()
  print('Fold: '+str(cross)+' Trained...')
  #model.evaluate(train_data, train_labels)
  #print('acc: '+str(history.history['accuracy'][-1]))# + '  val_acc:'+ str(history.history['val_acc'][-1]))
  preds=model.predict(validation_data)
  preds2=[]
  for i in range(len(preds)):
      r=np.round(preds[i])
      preds2.append(r)
  
  report=classification_report(validation_labels, preds2)
  print(report)
  accuracy.append(float(report.split()[15]))
  print('--------------------------')

print("All: Average ACC:"+str((np.sum(accuracy)/5)))
print("All:50 epoch ResNet50")