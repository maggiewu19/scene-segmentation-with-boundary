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
    return cv2.Canny(img, 200, 200)

def loopPixels(edges, image): 
    def updateColors(x, y, color): 
        results = set() 
        for i in range(x-1, x+1): 
            for j in range(y-1, y+1): 
                if 0 <= i < len(image) and 0 <= j < len(image[0]):
                    results.add((i, j, color))

        return results 

    global_results = set() 
    for index, pixel in np.ndenumerate(edges):
        x, y = index
        if pixel == 255: 
            color = tuple(image[x, y])
            global_results.update(updateColors(x, y, color))

    return global_results

def updateImage(filename):
    img = loadImage(filename)
    edges = getEdges(img)

    results = loopPixels(edges, img)
    for (i, j, color) in results: 
        img[i,j] = color

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

    test = loadImage(IMG_DIR + 'test.png')
    new_test = updateImage(IMG_DIR + 'test.png')

    room = loadImage(IMG_DIR + 'room.png')
    new_room = updateImage(IMG_DIR + 'room.png')

    saveImage('new_shells.png', new_shells)
    saveImage('new_plane.png', new_plane)
    saveImage('new_test.png', new_test)
    saveImage('new_wine.png', new_wine)
    saveImage('new_room.png', new_room)




