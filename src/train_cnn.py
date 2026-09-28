import cv2
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization #,CuDNNGRU
from keras.utils import to_categorical

import os
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
import matplotlib.pyplot as plt
from tkinter import filedialog
import requests


class Variable:
    try:
        def __init__(self):
            self.path_inputs = self.get_path_inputs()
            self.time = self.get_time()

        def get_path_inputs(self):
            return filedialog.askdirectory()

        def get_path_outputs_model(self):
            return f"/Users/deep03/Desktop/tle/Ricescan/Models/Model_{self.path_inputs.split('/')[-1]}_{self.time}.keras"
        
        def class_text_to_number(self, y):
            return list_class.index(y)
            
        def get_path_outputs_confu(self):
            return f"/Users/deep03/Desktop/tle/Ricescan/Outputs/Confusion_Matrix/confusion_{self.path_inputs.split('/')[-1]}_{self.time}.png"

        def get_path_outputs_loss(self):
            return f"/Users/deep03/Desktop/tle/Ricescan/Outputs/Curve_Loss/loss_{self.path_inputs.split('/')[-1]}_{self.time}.png"
        
        def get_time(self):
            return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


    except Exception as e:
        print(e)

var = Variable()

list_class = ['BN','HR','WG']

def load_image():
    X_list=[]
    y_list=[]

    arr = os.listdir(var.path_inputs)

    # list_img = [i for i in arr if i.startswith("UM") or i.startswith("DM") or i.startswith("SG")]
    list_img = [i for i in arr if not(i.startswith("HSV"))]

    for name_img in list_img:
        img = cv2.imread(f'{var.path_inputs}/{name_img.strip()}')

        ret,img_thres = cv2.threshold(img,35,255,cv2.THRESH_TOZERO)

        img_hsv = cv2.cvtColor(img_thres, cv2.COLOR_BGR2HSV_FULL)
        
        img_rot = img_hsv.copy()

        for i in range(1,3):
            save_img_hsv = cv2.imwrite(f'{var.path_inputs}/HSV_{i}_{name_img.strip()}', img_rot)

            load_img_hsv = load_img(f'{var.path_inputs}/HSV_{i}_{name_img.strip()}',target_size=(300,150))
            img_array = img_to_array(load_img_hsv)

            os.remove(f'{var.path_inputs}/HSV_{i}_{name_img.strip()}')

            X_list.append(img_array)
            y_list.append(int(var.class_text_to_number(name_img.strip().split('_')[0])))

            img_rot = cv2.rotate(img_rot, cv2.ROTATE_180)

    return X_list, y_list

def transfrom_data():

    X_list, y_list = load_image()

    X_input, y_input = np.array(X_list), np.array(y_list)


    X = X_input/255
    y = keras.utils.to_categorical(y_input, num_classes=len(set(y_input)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)
    # print(X_test)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, X_input, y_input

def model1(): #val_accuracy: 0.9535

    X_train, X_test, y_train, y_test, X_input, y_input = transfrom_data()
    start_time = datetime.now()

    model = Sequential()
    
    model.add(Conv2D(400, (4, 4), input_shape = (X_train.shape[1], X_train.shape[2], 3), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(600, (4, 4), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(800, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(1600, (4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(1000, (4, 4), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(800, (2, 2), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # model.add(Dense(units = 8192, activation = 'relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(units = 4096, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 2048, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 1024, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units = 256, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dropout(0.2))

    # model.add(Dense(units = 64 , activation= 'relu'))

    # model.add(Dense(units = 32, activation= 'relu'))

    # model.add(Dense(units = 16, activation= 'relu'))

    model.add(Dense(units = len(set(y_input)), activation = 'softmax'))

    model.compile(optimizer = Adam(learning_rate=0.0000001), loss='categorical_crossentropy', metrics = ['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',    # ตรวจสอบค่าความสูญเสียของ validation set
        patience=300,            # รอ 5 epochs ถ้า val_loss ไม่ดีขึ้น
        restore_best_weights=True  # กลับไปใช้ weight ที่ดีที่สุด
    )

    model.summary()

    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    # score = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=100000000, validation_data = [X_test, y_test], validation_split=0.2, callbacks=[early_stop])
    score = model.fit(X_train, y_train, epochs=100000000, validation_data = [X_test, y_test], validation_split=0.2, callbacks=[early_stop])

    loss, accuracy = model.evaluate(X_test, y_test)

    start_time_pred = datetime.now()
    pred = model.predict(X_test)
    
    predicted_class = np.argmax(pred, axis=1)

    # predicted_class = keras.utils.to_categorical(np.argmax(pred, axis=1), num_classes=len(set(np.argmax(pred, axis=1))))
    
    y_original = np.argmax(y_test, axis=1)
    print('Test accuracy:', accuracy)
    print(y_original, predicted_class, list_class)
    
    end_time = datetime.now()
    print(f"time = {end_time-start_time}")
    print(f'time predict = {end_time-start_time_pred}')

    requests.post("https://ntfy.sh/Notification_Train_Models", 
                  data=f"Notification from M2Ultra: Model {var.path_inputs.split('/')[-1]}\nModel time: {var.time}\nAccuracy: {round((accuracy*100),2)}\nTotal parameters: {model.count_params():,}\nTrain time: {end_time-start_time}")

    res = confusion_matrix(y_original, predicted_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=res,
                                display_labels=list_class)
    disp.plot()
    plt.savefig(var.get_path_outputs_confu(), dpi=300, bbox_inches='tight')
    # plt.show()

    plt.close()

    plt.plot(score.history['loss'])
    plt.plot(score.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(var.get_path_outputs_loss(), dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()

    #save model
    model.save(var.get_path_outputs_model())

    return True

if __name__ == "__main__":

    model1()

    # print(var.path_inputs)