import numpy as np
import cv2
from matplotlib import pyplot as plt

IMG_DIR = '../images/'

def plotImage(original, updated): 
    plt.subplot(121),plt.imshow(original, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(updated, cmap = 'gray')
    plt.title('Updated Image'), plt.xticks([]), plt.yticks([])

    plt.show()

image = cv2.imread(IMG_DIR + 'wine.png')
img = cv2.imread(IMG_DIR + 'wine.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
ret,thresh = cv2.threshold(gray,127,255,1)
 
img2, contours,h = cv2.findContours(thresh,1,2)
 
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt,True), True)
    print (len(approx))
    if len(approx)==4:
        print ("rectangle")
        cv2.drawContours(img,[cnt], 0, (255, 0, 0), -1)

plotImage(image, img)