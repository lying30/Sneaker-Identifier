# A program to identifier the type of sneaker from an image or video


from PIL import Image
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

TRAIN_DIR = './OFFWHITEYEEZY/TRAIN'
TEST_DIR = './OFFWHITEYEEZY/TEST'

IMG_SIZE = 120
LR = 1e-3

MODEL_NAME = 'OFFWHITEvsYEEZY--{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    # Images are formatted as: ADIDAS_1, NIKE_3 ...
    word_label = img.split('_')[0]
    if word_label == 'OFFWHITE': return [1,0] #one hot encoding
    elif word_label == 'YEEZY': return [0,1] #one hot encoding

