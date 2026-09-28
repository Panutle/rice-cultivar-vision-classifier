import cv2
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# import tensorflow_addons as tfa 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras import backend as K

from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization #,CuDNNGRU
from keras.utils import to_categorical

# from imblearn.over_sampling import SMOTE

import os
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
import matplotlib.pyplot as plt
from tkinter import filedialog

class Variable:
    try:
        def __init__(self):
            self.path_img = self.get_path_img()
            # self.model = self.get_model()
            self.path_model = self.get_path_model()

        def get_path_img(self):
            return filedialog.askopenfilenames(filetypes=[("JPEG files", "*.jpg *.png")])[0]

        def get_path_save_ss(self, i):
            return f"/Users/deep03/Desktop/tle/Ricescan/Demo_Project_3/{self.path_img.split('/')[-1].split('.')[0]}_{i+1}.jpg"

        def get_path_save_er(self, i):
            return f"/Users/deep03/Desktop/tle/Ricescan/Demo_Project_3_Error/{self.path_img.split('/')[-1].split('.')[0]}_{i+1}.jpg"
        
        # def get_model(self):
        #     return keras.models.load_model('/Users/deep03/Desktop/tle/Ricescan/Models/Model_BN_HR_WG_1.keras')
        
        def get_output_path(self):
            return f"/Users/deep03/Desktop/tle/Ricescan/Outputs/Output_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
        
        def get_path_model(self):
            path = filedialog.askopenfilenames(filetypes=[("Model files", "*.keras")])[0]

            return keras.models.load_model(path)
        
    except Exception as e:
        print(f'#################### Error at Variable: {e} ####################')

var = Variable()

# print(var.model)

def get_contour():
    try:

        global img_1

        img_1 = cv2.imread(var.path_img)

        img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

        ret, thresh_1 = cv2.threshold(img_gray_1, 70, 255, cv2.THRESH_BINARY)

        kernel = np.ones((6,6), np.uint8)

        n_k = 2

        img_ero = cv2.erode(thresh_1, kernel=kernel, iterations=n_k)
        img_dila = cv2.dilate(img_ero, kernel=kernel, iterations=n_k)

        countour_1, hierarchy_1 = cv2.findContours(image=img_dila,
                                                mode= cv2.RETR_CCOMP,
                                                method=cv2.CHAIN_APPROX_NONE)
        return countour_1

    except Exception as e:
        print(f'#################### Error at get_contour: {e} ####################')

def get_infomation_image (box, i, padding_ratio):
    try:    
    
        pts = box.copy()
        rect = pts

        center = np.mean(rect, axis=0)

        expanded_rect = []
        for point in rect:
            vector = point - center
            expanded_point = point + vector * padding_ratio
            expanded_rect.append(expanded_point)
        expanded_rect = np.array(expanded_rect, dtype="float32")

        widthA = np.linalg.norm(expanded_rect[2] - expanded_rect[3])
        widthB = np.linalg.norm(expanded_rect[1] - expanded_rect[0])
        heightA = np.linalg.norm(expanded_rect[1] - expanded_rect[2])
        heightB = np.linalg.norm(expanded_rect[0] - expanded_rect[3])
        maxWidth = int(max(widthA, widthB))
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth, 0],
            [maxWidth, maxHeight],
            [0, maxHeight]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(expanded_rect, dst)
        warped = cv2.warpPerspective(img_1, M, (maxWidth, maxHeight))

        if warped.shape[1] > warped.shape[0]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        warped = cv2.rotate(warped, cv2.ROTATE_180)

        return warped, warped.shape[0], warped.shape[1]
    
    except Exception as e:
        print(f'#################### Error at get_infomation_image: {e} ####################')

def get_input_model(warped, n_img, height, width):
    try:
        img = np.zeros((350,150,3), dtype='uint8').copy()
        
        for i, x in enumerate(warped):
            for j, y in enumerate(warped[i]):
                img[int(i+((img.shape[0]-height)/2))][int(j+((img.shape[1]-width)/2))] = y

        ret,img_thres = cv2.threshold(img,35,255,cv2.THRESH_TOZERO)
        img_hsv = cv2.cvtColor(img_thres, cv2.COLOR_BGR2HSV_FULL)

        os.makedirs(var.get_path_save_ss(n_img).split('/')[0] , exist_ok=True)
        cv2.imwrite(var.get_path_save_ss(n_img), img_hsv)
        
        load_img_hsv = load_img(var.get_path_save_ss(n_img), target_size=(300,150))

        os.remove(var.get_path_save_ss(n_img))

        return img_to_array(load_img_hsv)
        
    except Exception as e:
        print(f'#################### Error at get_input_model: {e} ####################')

def split_img_and_save():
    try:
        data_img = []
        input_img = []
        data_outputs = []

        for i, cnt in enumerate(get_contour()):
            data = dict()

            area = cv2.contourArea(cnt)

            if area > 20000 or area < 2000:
                pass

            else:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box) 

                padding_ratio = 0.2
                warped, height, width = get_infomation_image(box, i, padding_ratio)

                data['id'] = i
                data['cnt'] = cnt
                data['warped'] = warped
                data['width'] = width
                data['height'] = height
                data['area'] = area

                data_img.append(data)
        
        for n, data in enumerate(data_img):
            # print(data)
            input = get_input_model(data['warped'], n, data['height'], data['width'])

            input_img.append(input)
            # break

        inputs = np.array(input_img)/255

        # print(inputs)

        models = var.path_model

        models.compile(run_eagerly=True)

        pred = models.predict(inputs)

        predicted_class = list(np.argmax(pred, axis=1))

        # print(type(predicted_class))

        for i, output in enumerate(predicted_class):
            data = data_img[i]
            data['output'] = output
            data_outputs.append(data)

        draw = img_1.copy()

        for n, datas in enumerate(data_outputs):
            # print(datas['output'])
            if datas['output'] == 1:
                colors = (255,0,0)            
            elif datas['output'] == 2:
                colors = (0,255,0)
            else:
                colors = (0,0,255)
                
            img_contour = cv2.drawContours(image=draw,
                                    contours=datas['cnt'],
                                    contourIdx=-1,
                                    color=colors,
                                    thickness=5,
                                    lineType=cv2.LINE_AA)
        
        #Output image
        cv2.imwrite(str(var.get_output_path()), img_contour)

    except Exception as e:
        print(f'#################### Error at split_img_and_save: {e} ####################')

if __name__ == "__main__":
    try:    
        start_time = datetime.now()

        split_img_and_save()

        end_time = datetime.now()

        print(f'Time: {end_time-start_time}')

    except Exception as e:
        print(f'#################### Error at main: {e} ####################')