import cv2
import numpy as np
import os

directory = './dataset/'
path = './to_rectified'

def car_detection():
    #use trained cars XML classifiers
    car_cascade = cv2.CascadeClassifier('cars.xml')
    images = []

    ind = 0
    for filename in sorted(os.listdir(directory)):
        rect = []
        print(ind)
        i = cv2.imread(os.path.join(directory, filename))
        if i is not None:

            if ind%2 == 0:
                gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                cars = car_cascade.detectMultiScale(gray, 1.1, 2, minSize=(210, 210))
                if len(cars) == 0:
                for (x, y, w, h) in cars:
                	i[y:y+w, x:x+h] = 0
        cv2.imwrite(os.path.join(path, str(ind)+'.jpg'), i)
        ind = ind + 1


car_detection()