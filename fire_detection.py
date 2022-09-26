import  pandas as pd
import  numpy as np
import glob,cv2
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image

# check the number of file
def check_imag_file(file_path='dataset\\normal\\*.*'):
    img_file = glob.glob(file_path)
    print(f"Total number in {file_path} is: {len(img_file)}")

# build the label for file
def build_label_filepath(data_path,label=0):
    file_list=[]
    file_iterator=glob.glob(data_path)
    for file in file_iterator:
        file_list.append([file,label])
    return file_list

# read the img file by cv
def preprocessing_image(filepath,resize_num=196):
    img = cv2.imread(filepath) #read
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
    img = cv2.resize(img,(resize_num,resize_num))  # resize
    img = img / 255 #scale
    return img

# create format data
def create_format_dataset(dataframe):
    X = []
    y = []
    for f, t in dataframe.values:
        X.append(preprocessing_image(f))
        y.append(t)

    return np.array(X), np.array(y)

# set model
def build_model(X):
    model = Sequential()

    model.add(Conv2D(128, (2, 2), input_shape=(X.shape[1], X.shape[2], 3), activation='relu'))
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(1, activation="sigmoid"))
    #print(model.summary())
    return model

def resnet_model(X):
  IMAGE_SIZE=[X.shape[1],X.shape[2]]
  resnet=ResNet50(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)
  for layer in resnet.layers:
    layer.trainable = False

dataset_path="./drive/MyDrive/Colab Notebooks/Fire-Detection/"
fire_path=dataset_path+'1/*.*'
normal_path=dataset_path+'0/*.*'
check_imag_file(fire_path)
check_imag_file(normal_path)

fire_file_list=build_label_filepath(normal_path,label=0)
normal_file_list=build_label_filepath(fire_path,label=1)
total_file_list=fire_file_list+normal_file_list
random.shuffle(total_file_list)
df_total_file = pd.DataFrame(total_file_list, columns=['files', 'target'])

X, y = create_format_dataset(df_total_file)
print(X.shape,y.shape)

model=build_model(X)

train_epochs=200
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=train_epochs, batch_size=32, callbacks=callbacks)

y_pred = model.predict(X_test)

y_pred_reshape = y_pred.reshape(-1)
fire_thresold=0.75
y_pred_reshape[y_pred_reshape<fire_thresold] = 0
y_pred_reshape[y_pred_reshape>=fire_thresold] = 1
y_pred_reshape = y_pred_reshape.astype('int')
print(classification_report(y_test, y_pred_reshape))

from tensorflow.keras.models import load_model

model.save('fire_detection_init_model.h5')