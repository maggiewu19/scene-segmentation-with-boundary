import cv2
import numpy as np
import time 
from matplotlib import pyplot as plt

IMG_DIR = '../images/'

# white: 255
# black: 0 

def plotImage(original, updated): 
    plt.subplot(121),plt.imshow(original, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(updated, cmap = 'gray')
    plt.title('Updated Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def loadImage(filename):
    return cv2.imread(filename, 1)

def getEdges(img):
    return cv2.Canny(img, 100, 200)

def loopPixels(edges, image): 
    def updateColors(x, y, color): 
        results = set() 
        for i in range(x-1, x+2): 
            for j in range(y-1, y+2): 
                if 0 <= i < len(image) and 0 <= j < len(image[0]):
                    results.add((i, j))

        return results 

    global_results = set() 
    for index, pixel in np.ndenumerate(edges):
        x, y = index
        if pixel == 255: 
            color = image[x, y] 
            global_results.update(updateColors(x, y, color))

    return global_results

def updateImage(filename):
    img = loadImage(filename)
    edges = getEdges(img)

    results = loopPixels(edges, img)
    for (i, j) in results: 
        img[i,j] = 0 

    return img 

def saveImage(filename, img):
    cv2.imwrite(IMG_DIR + filename, img)

if __name__ == '__main__': 
    shells = loadImage(IMG_DIR + 'shells.png')
    new_shells = updateImage(IMG_DIR + 'shells.png')

    plane = loadImage(IMG_DIR + 'plane.png')
    new_plane = updateImage(IMG_DIR + 'plane.png')

    wine = loadImage(IMG_DIR + 'wine.png')
    new_wine = updateImage(IMG_DIR + 'wine.png')

    test = loadImage(IMG_DIR + 'test.jpg')
    new_test = updateImage(IMG_DIR + 'test.jpg')

    saveImage('new_shells.png', new_shells)
    saveImage('new_plane.png', new_plane)
    saveImage('new_test.png', new_test)
    saveImage('new_wine.png', new_wine)



