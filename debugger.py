# Used this file to test which images in my dataset were not working for my program. 

import cv2
from tqdm import tqdm
import os

TRAIN_DIR = '/Users/Lucas_Ying/Desktop/ATCS/Final Project_SneakerIdentifier/Sneaker-Identifier/data/TestingData'


train_data = []
for img in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, img)
    try:
        pic = cv2.imread(path)
        # print(img)
        pic = cv2.resize(pic, (100, 100))
        cv2.imshow("Image", pic)
        cv2.waitKey(0)
    except:
        print("Nope: ", path)

