# Lucas Ying
# 5/8/24
# THIS IS NOT PART OF MY CODE SUBMISSION
# This was my stretch goal I was working on to get the photo of the image to test from the camera 
# and it would take a screnshot of the camera and test that image, however I did not finish this part fully


import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt



RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


model = pickle.load(open('trained_model.pickle', 'rb'))
IMG_SIZE = 120

class Game:
    def __init__(self):
        # Load video
        self.video = cv2.VideoCapture(1)
    
    def plot_test_predictions(self, model, test_data):
        fig = plt.figure()
        for num, data in enumerate(test_data[:10]):
            img_data = data[0].reshape(IMG_SIZE, IMG_SIZE, 1)  # Correct reshaping
            model_out = model.predict(np.expand_dims(img_data, axis=0))[0]  # Correct input format for prediction
            str_label = 'YEEZY' if np.argmax(model_out) == 1 else 'OFFWHITE'
            y = fig.add_subplot(3, 4, num+1)
            y.imshow(img_data.squeeze(), cmap='gray')  # Use squeeze to remove singleton dimensions
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()
        
    def run(self):

        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]


            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(frame, 1)
            cv2.imshow("Hand Tracking", image)
 
            
            # Break the loop if the user presses 'q'
            key = cv2.waitKey(60) & 0xFF
            if key == ord('p'):
                image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
                cv2.imwrite("data/shoe.png", image)
                test_data = []
                img = cv2.imread("data/shoe.png", cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                test_data.append([np.array(img, dtype=np.uint8), 0])  # Ensure consistent data type
                np.save('test_data.npy', np.array(test_data, dtype=object))  # Using dtype=object for mixed types
                
                self.plot_test_predictions(model, test_data)

            if key == ord('q'):
                print("Game terminated.")
                break

        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()