import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import pandas as pd
import cv2
from keras.models import load_model
import os

model = load_model('handEmo.h5')        #Loaded saved model

def get_emojis():       #function to get emojis from directory and saving in list
    emojis_folder = 'emojis/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        emojis.append(cv2.imread(emojis_folder+str(emoji) + '.png', -1))
    return emojis

def keras_predict(model, image):        #Function to predict character on live web cam input
    processed = keras_process_image(image)
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed)[0]
    #print(pred_probab)
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class+1

def keras_process_image(img):           #Process image from live web cam input
    image_x = 50
    image_y = 50
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float64)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img

def overlay(image, emoji, x, y, w, h):      #function to overlay the emoji with the web cam feed
    emoji = cv2.resize(emoji, (w,h))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    try:
        image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image

def blend_transparent(background, foreground):      #function that performs the overlay
    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
                                alpha_background * background[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255
    return background
emojis = get_emojis()

cap = cap = cv2.VideoCapture(0)
x, y, w, h = 300, 50, 350, 350

while (cap.isOpened()):     #reading input from web cam
    ret, img = cap.read()
    img = cv2.flip(img,1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([108, 23, 82]), np.array([179, 255, 255]))
    res = cv2.bitwise_and(img, img, mask=mask2)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    median = cv2.GaussianBlur(gray, (5,5), 0)

    kernel_square = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(median, kernel_square, iterations=2)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)

    ret, thresh = cv2.threshold(opening, 30, 255, cv2.THRESH_BINARY)
    thresh = thresh[y:y+h, x:x+w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 2500:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            new_img = thresh[y1:y1+h1, x1:x1+w1]
            new_img = cv2.resize(new_img, (50, 50))     #fitting input into the model to get predicted output
            pred_probab, pred_class = keras_predict(model, new_img)
            print(pred_class, pred_probab)
            img = overlay(img, emojis[pred_class], 400, 250, 90, 90)       #overlaying predicted output to the web cam feed

    x, y, w, h = 300, 50, 350, 350
    cv2.imshow("Frame", img)
    cv2.imshow("Contours", thresh)      #Displaying web cam fed and contours
    k = cv2.waitKey(10)
    if k==27:       #exit program on escape input
        break