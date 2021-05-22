from matplotlib.image import imread
import imageio
import numpy as np
import pandas as pd
import os
import cv2
root = "./gestures_all"
count = 0;
for directory, subdirectories, files in os.walk(root):      #looping through gestures directory
    for file in files:
        print(file)
        im = imread(os.path.join(directory, file))         #reading image from their respective folders
        value = im.flatten()                #flattening matrix to vector
        value = np.hstack((directory[0:], value))
        df = pd.DataFrame(value).T          #converting vectors to pandas dataframe
        df = df.sample(frac=1)
        with open('train_foo.csv', 'a') as dataset:     #adding dataframes to csv file
            df.to_csv(dataset, header = False, index = False)
