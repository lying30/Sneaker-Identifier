# Lucas Ying Final Project
# A program to identifier the type of sneaker from an image or video

import cv2
from tqdm import tqdm
import os

TRAIN_DIR = '/Users/Lucas_Ying/Desktop/ATCS/Final Project_SneakerIdentifier/Sneaker-Identifier/data/TrainingData'


train_data = []
for img in tqdm(os.listdir(TRAIN_DIR)):
    path = os.path.join(TRAIN_DIR, img)
    # print(path)
    try:
        img = cv2.imread(path)
        cv2.imshow(img)
    except:
        print("Nope: ", path)

# img = cv2.imread("data/TestingData/OFFWHITE_1.jpg")
# cv2.imshow("Image", img)