#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2


def detect_plate(img):
    sample = img.copy()
    plate_rect = plate_cascade.detectMultiScale(sample,scaleFactor=1.2)

    #returns the coordinates of detected objects

    if len(plate_rect)==0:
        return "No plates detected"

    else:
        for (x,y,w,h) in plate_rect:
            sample[y:y+h,x:x+w] = cv2.GaussianBlur(sample[y:y+h,x:x+w],(99,99),0)

        # blurring every detected object

        return sample


plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")



pic = cv2.imread('PATH')

try :
    res = detect_plate(pic)
    if type(res) == str:
        print(res)
    else:
        plt.imshow(res)
        plt.show()
except AttributeError or ValueError:
    print("Invalid path or image")

    #try correcting your image path or maybe the image is corrupt